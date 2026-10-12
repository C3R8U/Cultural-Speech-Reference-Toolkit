"""Paired seed/sentence bootstrap and repeated-measures rating analyses."""
import argparse,csv,json,itertools
from pathlib import Path
import numpy as np
from scipy import stats

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid p-values')
    order=np.argsort(p); adjusted=np.empty(len(p));maximum=0.
    for i,k in enumerate(order):
        maximum=max(maximum,(len(p)-i)*p[k]);adjusted[k]=min(1.,maximum)
    return adjusted.tolist()

def mean_ci(values):
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all():raise ValueError('Need at least two finite independent seed results')
    mean=float(x.mean());sd=float(x.std(ddof=1));half=float(stats.t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return {'mean':mean,'sd':sd,'ci95':[mean-half,mean+half],'n_seeds':len(x)}

def paired_bootstrap(errors_a,errors_b,reference_counts,iterations=10000,seed=2026):
    a,b,n=[np.asarray(x,dtype=float) for x in (errors_a,errors_b,reference_counts)]
    if a.ndim!=2 or a.shape!=b.shape or a.shape!=n.shape:raise ValueError('Expected matching seed x sentence arrays')
    if a.shape[0]<2 or a.shape[1]<2:raise ValueError('Insufficient seed/sentence units')
    if any(not np.isfinite(x).all() for x in (a,b,n)) or (a<0).any() or (b<0).any() or (n<=0).any():
        raise ValueError('Invalid error/reference counts')
    if not np.all(n==n[0]):raise ValueError('Reference counts must be identical across seeds')
    if iterations<100:raise ValueError('At least 100 bootstrap iterations required')
    observed=float(np.mean(a.sum(1)/n.sum(1)-b.sum(1)/n.sum(1)))
    rng=np.random.default_rng(seed);draws=np.empty(iterations)
    for i in range(iterations):
        ss=rng.integers(a.shape[0],size=a.shape[0]);uu=rng.integers(a.shape[1],size=a.shape[1])
        ix=np.ix_(ss,uu);den=n[ix].sum(1)
        draws[i]=np.mean(a[ix].sum(1)/den-b[ix].sum(1)/den)
    # Null-centred two-sided bootstrap. Do not confuse tail mass around zero with a null test.
    p=(1+int(np.count_nonzero(np.abs(draws-observed)>=abs(observed))))/(iterations+1)
    return {'difference_a_minus_b':observed,'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'p_two_sided_null_centered':p,'iterations':iterations,'rng_seed':seed,
            'resampling':'paired crossed seed and sentence draws, identical across both systems'}

def load_error_grid(rows,model):
    selected=[r for r in rows if r['model']==model]
    if not selected:raise ValueError('Missing model '+model)
    cells={}
    for r in selected:
        key=(str(r['seed']),r['sample_id'])
        if key in cells:raise ValueError('Duplicate model/seed/sample record')
        cells[key]=r
    seeds=sorted({k[0] for k in cells});samples=sorted({k[1] for k in cells})
    if len(cells)!=len(seeds)*len(samples):raise ValueError('Incomplete seed x sentence grid')
    errors=np.array([[int(cells[(s,u)]['errors']) for u in samples] for s in seeds])
    counts=np.array([[int(cells[(s,u)]['reference_units']) for u in samples] for s in seeds])
    return seeds,samples,errors,counts

def analyze_errors(rows,baseline,iterations=10000):
    units={r['unit'] for r in rows}
    if len(units)!=1:raise ValueError('Mixed CER/WER units')
    models=sorted({r['model'] for r in rows})
    sa,ua,a,n=load_error_grid(rows,baseline);results=[]
    for model in models:
        if model==baseline:continue
        sb,ub,b,nb=load_error_grid(rows,model)
        if sa!=sb or ua!=ub or not np.array_equal(n,nb):raise ValueError('Models do not have identical paired units')
        references={(str(r['seed']),r['sample_id']):r.get('reference') for r in rows if r['model']==baseline}
        if any(references[(str(r['seed']),r['sample_id'])]!=r.get('reference') for r in rows if r['model']==model):
            raise ValueError('Reference text mismatch')
        result=paired_bootstrap(a,b,n,iterations);result['comparison']=baseline+' vs '+model;results.append(result)
    corrected=holm([r['p_two_sided_null_centered'] for r in results])
    for r,p in zip(results,corrected):r['p_holm']=p
    summary={}
    for model in models:
        _,_,e,c=load_error_grid(rows,model);summary[model]=mean_ci(e.sum(1)/c.sum(1))
    return {'unit':next(iter(units)),'seed_ids':sa,'sentence_count':len(ua),
            'seed_summaries':summary,'comparisons':results,
            'limitation':'Sentence bootstrap follows manuscript units; consider speaker-cluster sensitivity analysis'}

def analyze_ratings(rows):
    # Columns unit_id, model, score; pre-average raters within each output before calling.
    models=sorted({r['model'] for r in rows});units=sorted({r['unit_id'] for r in rows});cells={}
    for r in rows:
        k=(r['unit_id'],r['model'])
        if k in cells:raise ValueError('Duplicate unit/model; average raters explicitly first')
        value=float(r['score'])
        if not np.isfinite(value) or not 1<=value<=5:raise ValueError('Expected finite 1–5 ratings')
        cells[k]=value
    if len(models)<3 or len(units)<2 or len(cells)!=len(models)*len(units):raise ValueError('Incomplete repeated-measures grid')
    x=np.array([[cells[(u,m)] for m in models] for u in units]); pairs=[]
    if np.all(x==x[:,[0]]):stat,p=0.,1.
    else:
        f=stats.friedmanchisquare(*x.T);stat,p=float(f.statistic),float(f.pvalue)
    for i,j in itertools.combinations(range(len(models)),2):
        differences=x[:,i]-x[:,j]
        pw=1. if np.all(differences==0) else float(stats.wilcoxon(differences,zero_method='wilcox',method='approx').pvalue)
        pairs.append({'a':models[i],'b':models[j],'p':pw,'mean_paired_difference':float(differences.mean())})
    for r,pw in zip(pairs,holm([r['p'] for r in pairs])):r['p_holm']=pw
    return {'friedman_statistic':stat,'p':p,'kendall_w':stat/(len(units)*(len(models)-1)),
            'n_units':len(units),'models':models,'pairwise':pairs,
            'descriptives':{m:{'mean':float(x[:,i].mean()),'sd':float(x[:,i].std(ddof=1)),
                            'median':float(np.median(x[:,i]))} for i,m in enumerate(models)}}

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['errors','ratings']);p.add_argument('input')
    p.add_argument('--baseline',default='reference_dcsb');p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--out',required=True);a=p.parse_args()
    with open(a.input,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result=analyze_errors(rows,a.baseline,a.iterations) if a.kind=='errors' else analyze_ratings(rows)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
