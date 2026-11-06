"""R6c source/zero-aware mechanism reread, retaining every old counterexample.

State errors use actual H1, not old N0; labels and interpolation are references,
not inference inputs or physical truth. The hold14 filled-reference discrepancy
is retained. Raw/nonzero-pair ACF is not the sequence-demeaned published ACF.
"""
import argparse,time,traceback
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from peer20_population_core_v1_20261105 import *
from review_peer19_recipe_v3_20261104 import primary_state
from review_peer16_E4_E7_AP_v1_20261103 import geometry

OUT=BASE/'R6c';REVIEW=BASE/'R6c_independent'

def prepare():
    for name in ('R6c','R6c_independent'):
        register(name,__file__,__doc__,dict(GPU_forwards=0,CPU_threads=1,replicates=2000,disk_MiB='<20 sufficient stats/CDF, shared inputs'),
            roles=list(ROLES),R6a=ev(BASE/'R6a_raw_labels_independent_v2/terminal.json'),
            A1='Wait R6b EGO/PUBLIC/H1/OFFICIAL/MA5 full CI, consume separately; this task does A3/A4/A5 actual source reread',
            A5='raw vector ACF/nonzero-endpoint pairs1..100; also demeaned. First1/e crossing or right-censored; never make censored lag an exact tau.',
            independent_scope='Independent H1 matching/KF, independently aggregated state errors and ACF dot products; source joins/shared raw labels, descriptive model only')

def cluster_errors(seq,e,review=False):
    names=sorted(set(seq));groups=[]
    for s in names:
        v=e[seq==s];sq=np.einsum('ij,ij->i',v,v) if not review else np.array([float(np.dot(a,a)) for a in v])
        groups.append([len(v),float(sq.sum()),int((sq>1).sum())])
    g=np.asarray(groups,float)
    if not len(g):return dict(n=0,RMSE=None,gt1=None,CI95=None)
    rng=np.random.Generator(np.random.PCG64(2026110503));picks=rng.integers(0,len(g),(2000,len(g)));boot=g[picks].sum(1);v=g.sum(0)
    values=np.column_stack((np.sqrt(boot[:,1]/np.maximum(boot[:,0],1)),boot[:,2]/np.maximum(boot[:,0],1)))
    return dict(n=int(v[0]),RMSE=float(np.sqrt(v[1]/v[0])),gt1=float(v[2]/v[0]),CI95=np.quantile(values,[.025,.975],axis=0).T.tolist())

def correlation(m,ref,valid,demean,review):
    accum=np.zeros((100,5))
    for s in sorted(set(m['sequence'].astype(str))):
        ids=np.flatnonzero((m['sequence'].astype(str)==s)&(m['wait_ms']==0));ids=ids[np.argsort(m['receiver_timestamp_us'][ids],kind='stable')]
        value=ref[ids].copy();ok=valid[ids];times=m['receiver_timestamp_us'][ids].astype(float)/1e6
        if not ok.any():continue
        if demean:value-=value[ok].mean(0)
        for lag in range(1,min(101,len(ids))):
            use=ok[:-lag]&ok[lag:];a=value[:-lag][use];b=value[lag:][use];dt=(times[lag:]-times[:-lag])[use]
            if review:
                x=sum(float(np.dot(x,y)) for x,y in zip(a,b));xx=sum(float(np.dot(x,x)) for x in a);yy=sum(float(np.dot(y,y)) for y in b)
            else:x=np.sum(a*b);xx=np.sum(a*a);yy=np.sum(b*b)
            accum[lag-1]+=[x,xx,yy,float(dt.sum()),int(use.sum())]
    rho=accum[:,0]/np.maximum(np.sqrt(accum[:,1]*accum[:,2]),1e-15);seconds=accum[:,3]/np.maximum(accum[:,4],1);good=accum[:,4]>0
    cross=np.flatnonzero(good&(rho<=np.exp(-1)))
    return dict(rho=rho.tolist(),pair_counts=accum[:,4].astype(int).tolist(),lag_seconds=seconds.tolist(),demeaned=demean,
        tau_c_seconds=float(seconds[cross[0]]) if len(cross) else None,right_censored=not bool(len(cross)),
        censor_lower_s=float(seconds[np.flatnonzero(good)[-1]]) if good.any() else None)

def calculate(review=False):
    dataset,_,_=data('author');cal=load(REC/'author/calibration.json');q=cal['q'];state_rows=[];acf=[];cdf=[]
    for role in ROLES:
        m,rows=dataset[role];pops,off=annotations(rows);filled,ma,span,supported=fill(rows,off)
        if review:d,n,z=primary_state(m,q)
        else:
            st=h1.recipe.hard_state(m,q);d,n,z=st['delta'],st['anchors'],st['observation']
        seq=m['sequence'].astype(str)
        for w in WAITS:
            for reference,target,mask in [('official_nonzero',off,pops['nonzero']),('filled',filled,np.ones(len(rows),bool)),('filled_MA5',ma,np.ones(len(rows),bool))]:
                for pop in ('all','zero','colabelled','nonzero','anchored','anchor_free'):
                    popmask=(n>0 if pop=='anchored' else n==0 if pop=='anchor_free' else pops[pop]);use=mask&popmask&(m['wait_ms']==w)
                    for method,value in [('H1',d),('single_message',z),('zero_fallback',np.zeros_like(d))]:
                        state_rows.append(dict(role=role,wait_ms=w,reference=reference,population=pop,method=method,
                            **cluster_errors(seq[use],(value-target)[use],review)))
        for ref,target,mask in [('official_nonzero_pairs',off,pops['nonzero']),('filled',filled,np.ones(len(rows),bool))]:
            for demean in (False,True):
                result=correlation(m,target,mask,demean,review);w0=m['wait_ms']==0;observed=(n>0)&w0
                duration=sum(max((m['receiver_timestamp_us'][(seq==s)&w0].max()-m['receiver_timestamp_us'][(seq==s)&w0].min())/1e6,.1) for s in sorted(set(seq)))
                rate=float(observed.sum()/duration);fraction=float(observed.sum()/w0.sum());mbar=float(n[observed].mean())
                Q=q*q;R=.36/mbar;prior=(Q+np.sqrt(Q*Q+4*fraction*Q*R))/(2*fraction);pred=float(np.sqrt(2*max(prior-Q,0)))
                use=w0&mask;measured=float(np.sqrt(np.mean(np.sum((d[use]-target[use])**2,axis=1))))
                result.update(role=role,reference=ref,q=q,lambda_informative_Hz=rate,mean_anchors_given_informative=mbar,
                    N_eff=(result['tau_c_seconds']*rate if result['tau_c_seconds'] is not None else None),
                    N_eff_lower_bound=(result['censor_lower_s']*rate if result['right_censored'] else None),
                    predicted_RMSE=pred,measured_RMSE=measured,relative_prediction_error=abs(pred-measured)/measured,
                    main_text_numeric_bound_authorized=False)
                acf.append(result)
        code,_=codes(m,rows,role);before=geometry(m,np.zeros_like(d));after=geometry(m,d);groups=defaultdict(list)
        for i,w in enumerate(m['wait_ms']):
            a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2]);remote=a+np.flatnonzero(m['source'][a:b]==1)
            for source,label in enumerate(('veh','coop','inf')):
                gt=ga+np.flatnonzero(code[ga:gb]==source)
                if not len(gt) or not len(remote):continue
                dist=np.linalg.norm(before[remote,None,:2]-m['gt_boxes'][None,gt,:2],axis=-1);x,y=linear_sum_assignment(dist);keep=dist[x,y]<=3.
                for ri,gi in zip(remote[x[keep]],gt[y[keep]]):
                    groups[int(w),label,bool(pops['zero'][i])].append((str(seq[i]),before[ri,:2]-m['gt_boxes'][gi,:2],after[ri,:2]-m['gt_boxes'][gi,:2]))
        for (w,source,zero),values in sorted(groups.items()):
            ss=np.array([v[0] for v in values]);old=np.array([v[1] for v in values]);new=np.array([v[2] for v in values])
            cdf.append(dict(role=role,wait_ms=w,source=source,zero=zero,fixed_matches=len(values),
                PUBLIC=cluster_errors(ss,old,review),H1=cluster_errors(ss,new,review),
                CDF={k:dict(threshold_m=[.1,.25,.5,1.,2.],fraction=[float(np.mean(np.linalg.norm(e,axis=1)<=t)) for t in (.1,.25,.5,1.,2.)]) for k,e in [('PUBLIC',old),('H1',new)]}))
    return dict(A3=state_rows,A4=cdf,A5=acf,official_GT_unchanged=True,
        A1_pending_R6b=True,scope='Direct source/zero strata; matched-centre CDF is not all-GT recall; filled references label-assisted; right-censored ACF is not exact tau.')

def execute(review=False):
    name='R6c_independent' if review else 'R6c';dest=REVIEW if review else OUT
    try:
        if review:wait(OUT/'terminal.json')
        record(name,dict(status='running_mechanism_reread',resource_owner=True));result=calculate(review)
        if review:
            gold=load(OUT/'mechanism.json')
            def same(a,b):
                if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
                if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
                if isinstance(a,float):return abs(a-b)<1e-7
                return a==b
            assert same(result,gold),'independent mechanism mismatch'
        atomic(dest/'mechanism.json',result);atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
            mechanism=ev(dest/'mechanism.json'),A1_completion_not_implied=True))
        record(name,dict(status=load(dest/'terminal.json')['status'],terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));a=p.parse_args();prepare() if a.mode=='prepare' else execute(a.mode=='review')
