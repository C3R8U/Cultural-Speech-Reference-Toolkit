"""CTC prefix beam search; path probabilities merged before semantic reranking."""
import math
import torch
from torch.nn import functional as F

def logadd(*values):
    m = max(values)
    return m if m == -math.inf else m + math.log(sum(math.exp(v-m) for v in values))

def prefix_beam(log_probs, beam_size=10, blank=0):
    beams = {(): (0., -math.inf)}
    for frame in log_probs:
        next_beams = {}
        def add(prefix, pb=-math.inf, pn=-math.inf):
            if pb == -math.inf and pn == -math.inf: return
            old = next_beams.get(prefix, (-math.inf,-math.inf))
            next_beams[prefix] = (logadd(old[0],pb),logadd(old[1],pn))
        for prefix, (pb,pn) in beams.items():
            add(prefix, pb=logadd(pb,pn)+float(frame[blank]))
            for k, p in enumerate(frame):
                if k == blank or k in (1,2): continue
                p = float(p)
                if prefix and k == prefix[-1]:
                    add(prefix, pn=pn+p)
                    add(prefix+(k,), pn=pb+p)
                else:
                    add(prefix+(k,), pn=logadd(pb,pn)+p)
        beams = dict(sorted(next_beams.items(), key=lambda item:logadd(*item[1]), reverse=True)[:beam_size])
    return [(list(p),logadd(*scores)) for p,scores in sorted(beams.items(),key=lambda item:logadd(*item[1]),reverse=True)]

@torch.no_grad()
def rerank(model, representation, candidates, alpha=.1, top_n=5):
    ranked = []
    for sequence, ctc_score in candidates[:top_n]:
        if sequence:
            ids = torch.tensor([sequence],device=representation.device)
            semantic = model.semantic_representation(model.characters(ids),torch.tensor([len(sequence)],device=ids.device))
            similarity = float(F.cosine_similarity(semantic,representation)[0])
        else: similarity = 0.
        ranked.append((sequence, ctc_score+alpha*similarity, ctc_score, similarity))
    return sorted(ranked,key=lambda x:x[1],reverse=True)
