"""MFCC/BiLSTM/CTC, blank-aware bridge, semantic LSTM and generator."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

@dataclass
class Config:
    vocab_size: int
    hidden: int = 256
    embedding: int = 128
    acoustic_layers: int = 3
    semantic_layers: int = 2
    dropout: float = .30
    bridge: str = 'dcsb'  # dcsb, soft, hard

def run_lstm(module, sequence, lengths):
    packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, state = module(packed)
    output, _ = pad_packed_sequence(output, batch_first=True, total_length=sequence.shape[1])
    return output, state

class SpeechSemanticModel(nn.Module):
    # 0=CTC blank/padding, 1=BOS, 2=EOS, 3=UNK; all text tokens >=3.
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.acoustic = nn.LSTM(39, config.hidden, config.acoustic_layers,
            batch_first=True, bidirectional=True, dropout=config.dropout if config.acoustic_layers > 1 else 0)
        self.ctc_projection = nn.Linear(config.hidden*2, config.vocab_size)
        self.characters = nn.Embedding(config.vocab_size, config.embedding, padding_idx=0)
        self.semantic = nn.LSTM(config.embedding, config.hidden, config.semantic_layers,
            batch_first=True, dropout=config.dropout if config.semantic_layers > 1 else 0)
        self.intent = nn.Linear(config.hidden, 3)
        self.decoder = nn.LSTM(config.embedding, config.hidden, 2, batch_first=True, dropout=config.dropout)
        self.generation_projection = nn.Linear(config.hidden, config.vocab_size)

    def semantic_representation(self, embeddings, lengths, weights=None):
        hidden, _ = run_lstm(self.semantic, embeddings, lengths)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None,:] < lengths[:,None]
        w = mask.to(hidden.dtype) if weights is None else weights * mask
        return (hidden*w[:,:,None]).sum(1)/w.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, features, lengths, decoder_input=None):
        hidden, _ = run_lstm(self.acoustic, features, lengths)
        logits = self.ctc_projection(hidden)
        # BOS/EOS are reserved decoder symbols, not acoustic labels.
        logits = logits.clone()
        logits[:,:,1:3] = -1e4
        posterior = logits.softmax(-1)
        confidence = (1-posterior[:,:,0]).clamp_min(0)
        if self.config.bridge == 'hard':
            tokens = []
            for row, n in zip(posterior.argmax(-1), lengths):
                previous, seq = None, []
                for k in row[:int(n)].tolist():
                    if k != previous and k != 0: seq.append(k)
                    previous = k
                tokens.append(seq or [3])
            token_lengths = torch.tensor([len(s) for s in tokens], device=features.device)
            padded = torch.zeros(len(tokens), int(token_lengths.max()), dtype=torch.long, device=features.device)
            for i, seq in enumerate(tokens): padded[i,:len(seq)] = torch.tensor(seq, device=features.device)
            representation = self.semantic_representation(self.characters(padded), token_lengths)
        elif self.config.bridge == 'soft':
            embeddings = posterior @ self.characters.weight
            representation = self.semantic_representation(embeddings, lengths)
        elif self.config.bridge == 'dcsb':
            # Blank excluded from semantic distribution; nonblank confidence gates input and pooling.
            distribution = posterior[:,:,1:]/confidence[:,:,None].clamp_min(1e-8)
            embeddings = (distribution @ self.characters.weight[1:]) * confidence[:,:,None]
            representation = self.semantic_representation(embeddings, lengths, confidence)
        else:
            raise ValueError('Unknown bridge')
        result = {'ctc_logits': logits, 'intent_logits': self.intent(representation),
                  'representation': representation}
        if decoder_input is not None:
            h0 = representation.unsqueeze(0).repeat(2,1,1).contiguous()
            decoded, _ = self.decoder(self.characters(decoder_input), (h0, torch.zeros_like(h0)))
            result['generation_logits'] = self.generation_projection(decoded)
        return result

    @torch.no_grad()
    def generate(self, representation, max_tokens=64, beam_size=5):
        if representation.shape[0] != 1: raise ValueError('Generation uses batch size 1')
        h = representation.unsqueeze(0).repeat(2,1,1).contiguous()
        beams = [(0., [1], (h, torch.zeros_like(h)), False)]
        for _ in range(max_tokens):
            candidates = []
            for score, sequence, state, ended in beams:
                if ended:
                    candidates.append((score, sequence, state, ended)); continue
                token = torch.tensor([[sequence[-1]]], device=representation.device)
                out, next_state = self.decoder(self.characters(token), state)
                logp = self.generation_projection(out[:,-1]).log_softmax(-1)[0]
                logp[0:2] = -float('inf')
                values, ids = logp.topk(min(beam_size, self.config.vocab_size-2))
                for v, k in zip(values.tolist(), ids.tolist()):
                    candidates.append((score+v, sequence+[k], next_state, k==2))
            beams = sorted(candidates, key=lambda x:x[0], reverse=True)[:beam_size]
            if all(b[3] for b in beams): break
        return [k for k in beams[0][1][1:] if k != 2]

def joint_loss(output, batch, weights=(.60,.25,.15)):
    ctc = nn.CTCLoss(blank=0, zero_infinity=False)(
        output['ctc_logits'].log_softmax(-1).transpose(0,1), batch['ctc_targets'],
        batch['lengths'], batch['ctc_lengths'])
    valid = batch['intent_labels'] >= 0
    intent = F.cross_entropy(output['intent_logits'][valid], batch['intent_labels'][valid]) if valid.any() else ctc*0
    generation = ctc*0
    if 'generation_logits' in output and (batch['decoder_target'] != 0).any():
        generation = F.cross_entropy(output['generation_logits'].reshape(-1, output['generation_logits'].shape[-1]),
                                     batch['decoder_target'].reshape(-1), ignore_index=0)
    total = weights[0]*ctc + weights[1]*intent + weights[2]*generation
    if not torch.isfinite(total): raise ValueError('Nonfinite loss: check CTC lengths/repeated labels and data')
    return total, {'ctc':float(ctc.detach()), 'intent':float(intent.detach()), 'generation':float(generation.detach())}

def save_checkpoint(path, model, vocab, extra=None):
    torch.save({'config':asdict(model.config), 'state_dict':model.state_dict(), 'vocab':vocab,
                'extra':extra or {}, 'implementation':'new_reference_implementation'}, path)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = SpeechSemanticModel(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint
