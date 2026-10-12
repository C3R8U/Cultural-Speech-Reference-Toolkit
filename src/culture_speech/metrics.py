import unicodedata

def edit_distance(reference, hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]

def error_counts(reference, hypothesis, unit='character'):
    if unit=='character':
        from .data import normalize
        reference,hypothesis=list(normalize(reference)),list(normalize(hypothesis))
    elif unit=='word':
        import jieba
        from .data import normalize
        reference,hypothesis=list(jieba.cut(normalize(reference))),list(jieba.cut(normalize(hypothesis)))
    else: raise ValueError('Unknown unit')
    if not reference: raise ValueError('Empty reference')
    return edit_distance(reference,hypothesis),len(reference)

def generation_metrics(references,hypotheses):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score.rouge_scorer import RougeScorer
    from .data import normalize
    if not references or len(references)!=len(hypotheses): raise ValueError('Unpaired or empty generation data')
    refs=[list(normalize(x)) for x in references]; hyps=[list(normalize(x)) for x in hypotheses]
    if any(not x for x in refs): raise ValueError('Empty generation reference')
    class CharacterTokenizer:
        def tokenize(self,text): return list(normalize(text))
    scorer=RougeScorer(['rougeL'],tokenizer=CharacterTokenizer())
    return {'bleu2_character_0_100':100*corpus_bleu([[x] for x in refs],hyps,weights=(.5,.5),
                smoothing_function=SmoothingFunction().method1),
            'rougeL_character_0_100':100*sum(scorer.score(r,h)['rougeL'].fmeasure
                         for r,h in zip(references,hypotheses))/len(refs),
            'n':len(refs),'note':'Reference implementation tokenization; not manuscript-original scoring settings'}
