"""R6c/A5 full sequence-bootstrap CIs, raw and demeaned/nonzero-pair ACF.

No numerical-bound authorization, even if an alternate reference narrows the
gap. No-crossing draws are right-censored, not assigned exact tau. Sequence
means are removed within each original sequence, never across sequences.
"""
import argparse,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
from review_peer19_recipe_v3_20261104 import primary_state

def prepare():
    for review in (False,True):
        register('R6c_A5_cluster'+('_independent' if review else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<10 sufficient stats and summary'),
            references=['official_nonzero_pairs','filled'],demeaning=[False,True],max_message_lag=100,
            independence='Independent scalar dot products and primary KF versus production vectorized aggregation. Same raw labels, interpolation and bootstrap draws. No calibrated physical truth.')

def summary(a,g,counts):
    # a: sequence x100lags x5, g: sequence x7 sufficient scalars.
    pooled=np.einsum('bs,slk->blk',counts,a);v=counts@g
    den=np.sqrt(pooled[:,:,1]*pooled[:,:,2]);rho=pooled[:,:,0]/np.maximum(den,1e-15)
    dt=pooled[:,:,3]/np.maximum(pooled[:,:,4],1);good=pooled[:,:,4]>0
    fraction=v[:,0]/v[:,1];mbar=v[:,2]/np.maximum(v[:,0],1);rate=v[:,0]/v[:,3]
    q=v[:,6]/counts.sum(1);Q=q*q;R=.36/np.maximum(mbar,1e-15)
    prior=(Q+np.sqrt(Q*Q+4*fraction*Q*R))/(2*np.maximum(fraction,1e-15))
    pred=np.sqrt(2*np.maximum(prior-Q,0));measured=np.sqrt(v[:,4]/np.maximum(v[:,5],1));relative=abs(pred-measured)/np.maximum(measured,1e-15)
    tau=[];censor=[];neff=[]
    for i in range(len(counts)):
        crossing=np.flatnonzero(good[i]&(rho[i]<=np.exp(-1)))
        hit=bool(len(crossing));last=np.flatnonzero(good[i]);value=float(dt[i,crossing[0]]) if hit else None
        lower=float(dt[i,last[-1]]) if len(last) else None
        tau.append(value);censor.append(lower if not hit else None);neff.append(value*rate[i] if hit else None)
    return rho,dict(lambda_Hz=rate,mbar=mbar,predicted_RMSE=pred,measured_RMSE=measured,relative_prediction_error=relative),tau,censor,neff

def calculate(review):
    dataset,_,_=data('author');q=load(REC/'author/calibration.json')['q'];result=[]
    for role in ROLES:
        m,rows=dataset[role];pops,off=annotations(rows);filled=fill(rows,off)[0]
        if review:d,n,_=primary_state(m,q)
        else:st=h1.recipe.hard_state(m,q);d,n=st['delta'],st['anchors']
        seq=m['sequence'].astype(str);names=sorted(set(seq));rng=np.random.Generator(np.random.PCG64(2026110505));picks=rng.integers(0,len(names),(2000,len(names)))
        counts=np.asarray([np.bincount(r,minlength=len(names)) for r in picks],np.int16)
        for ref,target,valid in [('official_nonzero_pairs',off,pops['nonzero']),('filled',filled,np.ones(len(rows),bool))]:
            for demean in (False,True):
                a=np.zeros((len(names),100,5));g=np.zeros((len(names),7))
                for j,s in enumerate(names):
                    ids=np.flatnonzero((seq==s)&(m['wait_ms']==0));ids=ids[np.argsort(m['receiver_timestamp_us'][ids],kind='stable')]
                    ts=m['receiver_timestamp_us'][ids].astype(float)/1e6;v=target[ids].copy();ok=valid[ids]
                    if demean and ok.any():v-=v[ok].mean(0)
                    observed=n[ids]>0;err=d[ids][ok]-target[ids][ok]
                    g[j]=[observed.sum(),len(ids),n[ids][observed].sum(),max(ts[-1]-ts[0],.1),np.sum(err**2),ok.sum(),q]
                    for lag in range(1,min(101,len(ids))):
                        use=ok[:-lag]&ok[lag:];x,y=v[:-lag][use],v[lag:][use];dt=(ts[lag:]-ts[:-lag])[use]
                        if review:terms=[sum(float(np.dot(xx,yy)) for xx,yy in zip(x,y)),sum(float(np.dot(xx,xx)) for xx in x),sum(float(np.dot(yy,yy)) for yy in y)]
                        else:terms=[np.sum(x*y),np.sum(x*x),np.sum(y*y)]
                        a[j,lag-1]=[*terms,dt.sum(),use.sum()]
                rho,values,tau,censor,neff=summary(a,g,np.vstack((np.ones((1,len(names)),np.int16),counts)))
                def ci(v):return np.quantile(v[1:],[.025,.975]).tolist()
                finite_tau=[x for x in tau[1:] if x is not None];finite_neff=[x for x in neff[1:] if x is not None];lower=[x for x in censor[1:] if x is not None]
                result.append(dict(role=role,reference=ref,demeaned=demean,replicates=2000,sequences=len(names),rho=rho[0].tolist(),rho_CI95=np.quantile(rho[1:],[.025,.975],axis=0).T.tolist(),
                    statistics={k:dict(point=float(v[0]),CI95=ci(v)) for k,v in values.items()},tau_seconds=tau[0],right_censor_lower_seconds=censor[0],N_eff=neff[0],
                    censored_draws=len(lower),finite_crossing_draws=len(finite_tau),
                    finite_crossing_tau_CI_conditional=np.quantile(finite_tau,[.025,.975]).tolist() if finite_tau else None,
                    finite_crossing_Neff_CI_conditional=np.quantile(finite_neff,[.025,.975]).tolist() if finite_neff else None,
                    censor_lower_seconds_CI_conditional=np.quantile(lower,[.025,.975]).tolist() if lower else None,
                    CI_censoring_note='Conditional intervals are not unconditional tau bounds; no-crossing draws retained/counted',main_text_numeric_bound_authorized=False))
    return result

def run(review=False):
    name='R6c_A5_cluster'+('_independent' if review else '');dest=BASE/name;src=BASE/'R6c_A5_cluster'
    try:
        if review:wait(src/'terminal.json')
        record(name,dict(status='running_sequence_cluster_ACF_CI',resource_owner=True));r=calculate(review)
        if review:
            from run_peer20_mechanism_v1_20261105 import calculate as oldcalculate
            # Full independent result recursively agrees to numerical tolerance.
            def eq(a,b):
                if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
                if isinstance(a,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
                if isinstance(a,float):return abs(a-b)<1e-7
                return a==b
            assert eq(r,load(src/'panels.json'))
        atomic(dest/'panels.json',r);status='complete_independently_reviewed' if review else 'complete_light_checked'
        atomic(dest/'terminal.json',dict(created_at=now(),status=status,panels=ev(dest/'panels.json'),no_main_text_numeric_bound=True))
        record(name,dict(status=status,terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode=='review')
