"""Direct current-method comparisons after the independently accepted gate.

Use actual accepted selected-component predictions, not sums of old AP deltas.
Compare same-author native, frozen stronger-late columns, three same-readout
modules, and four actual native checkpoints. All source matches were already
independently accepted. New AP/sequence CI uses the independent arithmetic.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import review_peer18_final_native_statistics_v1_20261104 as ind
from review_peer18_main_module_statistics_v1_20261104 import pack
import bootstrap_peer17_main_contrasts_v1_20261104 as main
import run_peer17_module_recipe_v1_20261104 as module
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import materials
OUT=BASE/'accepted_component_direct_contrasts_v1'

def prepare():
    register('accepted_component_direct_contrasts_v1',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        panels=390,replicates=2000,wall_h='bounded26h includinggate',disk_MiB='<25 compact contrasts, shared source predictions'),
        gate=ev(BASE/'adoption_independent_v1/contract.json'),kernel=ev(ind.__file__),
        baseline_reviews=[ev(BASE/n/'terminal.json') for n in ('final_native_statistics_independent','main_module_statistics_independent')],
        invariant='Only a globally accepted component is compared. No new variant/fit/choice, no per-cell replacement.')

def stronger(m,p,th):
    seqs=sorted(set(m['sequence'].astype(str)));one=np.ones((1,len(seqs)),np.int16)
    d={n:pack(m,p,n,th) for n in ('official_late_no_comp','official_late_time_comp')}
    combined=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[]);columns={}
    for w in (0,100,200,400):
        aps={n:float(ind.empirical(*ind.panel(x,w,seqs),one)[0]) for n,x in d.items()}
        name=max(aps,key=lambda n:(aps[n],n));columns[str(w)]=name;x=d[name]
        for i in np.flatnonzero(x['wait']==w):
            lo,hi=map(int,x['offset'][i:i+2])
            for k in ('sequence','wait','gt'):combined[k].append(x[k][i])
            for k in ('score','tp'):combined[k].extend(x[k][lo:hi])
            combined['offset'].append(len(combined['score']))
    return {k:np.asarray(v) for k,v in combined.items()},columns

def contrast(a,b,label,th,identity,done):
    seqs=sorted(set(a['sequence'].astype(str)));assert seqs==sorted(set(b['sequence'].astype(str)))
    for s in seqs:
        for w in (0,100,200,400):
            assert np.array_equal(a['gt'][(a['sequence'].astype(str)==s)&(a['wait']==w)],b['gt'][(b['sequence'].astype(str)==s)&(b['wait']==w)]),(label,s,w)
    draws=np.random.Generator(np.random.PCG64(20261103)).integers(0,len(seqs),(2000,len(seqs)))
    counts=np.asarray([np.bincount(r,minlength=len(seqs)) for r in draws],np.int16)
    ds=[];points=[];perseq=[]
    for w in (0,100,200,400):
        x,y=ind.panel(a,w,seqs),ind.panel(b,w,seqs);assert np.array_equal(x[3],y[3])
        ds.append(100*(ind.empirical(*x,counts)-ind.empirical(*y,counts)))
        one=np.ones((1,len(seqs)),np.int16)
        points.append(float(100*(ind.empirical(*x,one)[0]-ind.empirical(*y,one)[0])))
        perseg=100*(ind.empirical(*x,np.eye(len(seqs),dtype=np.int16))-ind.empirical(*y,np.eye(len(seqs),dtype=np.int16)))
        perseq.append(perseg)
    np.savez_compressed(OUT/(label+'_AP'+str(th)+'.npz'),sequence=np.asarray(seqs),cluster_counts=counts,**{'wait'+str(w):v for w,v in zip((0,100,200,400),ds)})
    for w,pt,dd,ss in zip((0,100,200,400,'four_wait_mean'),points+[float(np.mean(points))],ds+[np.mean(ds,axis=0)],perseq+[np.mean(perseq,axis=0)]):
        ci=np.percentile(dd,[2.5,97.5]);done.append(dict(comparison=label,threshold=th,wait_ms=w,delta_pp=pt,CI95_low=float(ci[0]),CI95_high=float(ci[1]),
            positive_sequences=int((ss>1e-12).sum()),negative_sequences=int((ss< -1e-12).sum()),zero_sequences=int((abs(ss)<=1e-12).sum()),
            identity=identity,independent_statistics=True,saved_matching_shared=True))
    atomic(OUT/'panels.json',done);atomic(OUT/'progress.json',dict(created_at=now(),panels=len(done),comparison=label,threshold=th))

def current(domain,role,seed,variant):
    p=BASE/'CoMET_Net/formal'/('seed'+str(seed))/(domain+'__to__'+domain+'__'+variant)
    assert load(BASE/'CoMET_Net/independent'/('seed'+str(seed))/p.name/'terminal.json')['status']=='complete_independently_reviewed'
    return p/(role+'_predictions.npz')

def run():
    begin=time.monotonic();done=[];dep=BASE/'adoption_independent_v1/terminal.json'
    record('accepted_component_direct_contrasts_v1',dict(status='waiting_independent_component_gate',resource_owner=False,owner=None))
    try:
        while not dep.exists():assert time.monotonic()-begin<26*3600;time.sleep(30)
        assert load(dep)['status']=='complete_independently_reviewed'
        decision=load(dep)['recommendation']
        if decision=='retain_CoMET':
            atomic(OUT/'panels.json',[])
        else:
            v='N4' if decision=='CoMET_Net_N4' else load(BASE/'adoption_independent_v1/method_registry.json')['component']
            assert v in ('N2','N3','N4'),('No prior learned formal predictions for this component',v)
            record('accepted_component_direct_contrasts_v1',dict(status='running_direct_accepted_component_contrasts',resource_owner=True))
            data,_,_=materials('author');_,contracts=main.boot.configs()
            for role in ('val21','mechanism_holdout14'):
                m,mr=data[role];lm,lr,lp,_,_=main.boot.aux.official_late(role,contracts['official_late'])
                key=lambda rr:sorted((str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])) for r in rr)
                assert key(mr)==key(lr)
                for th in (50,70):
                    b,columns=stronger(lm,lp,th)
                    for s in final.SEEDS:
                        path=current('author',role,s,v)
                        with np.load(path) as p:a=pack(m,p,v+'_seed'+str(s),th)
                        identity=dict(role=role,domain='author',seed=s,variant=v,current=ev(path),N0_identity=ev(final.OUT/'author/identity.json'),baseline_columns=columns)
                        contrast(a,b,f'{role}_{v}_seed{s}_minus_stronger_late',th,identity,done)
                        if role=='val21':contrast(a,main.native(th),f'val21_{v}_seed{s}_minus_same_author_native',th,identity,done)
            _,cs=module.old.configs()
            for meth in module.MODULES:
                for role in ('val21','mechanism_holdout14'):
                    m,_,_,_,_=module.old.modules(role,cs['modules'])
                    for s in final.SEEDS:
                        path=current('author',role,s,v);base=module.OUT/meth/f'seed{s}'/(role+'_predictions.npz')
                        with np.load(path) as p,np.load(base) as q:
                            for th in (50,70):
                                contrast(pack(m,p,v+'_seed'+str(s),th),pack(m,q,meth+'_U_seed'+str(s),th),f'{role}_{v}_minus_{meth}_seed{s}',th,
                                    dict(domain='author',role=role,seed=s,variant=v,current=ev(path),baseline=ev(base),module_not_system=True),done)
            for d,src in ind.SOURCES.items():
                m,_=materials(d)[0]['val21']
                for th in (50,70):
                    if d!='where2comm':
                        with np.load(src/f'native_AP{th}_compact.npz') as z:b={k:z[k].copy() for k in z.files}
                    else:
                        b=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[])
                        for w in (0,100,200,400):
                            with np.load(src/'where'/f'common_PUBLIC_AP{th}_wait{w}.npz') as z:
                                for k in ('sequence','wait','gt','score','tp'):b[k].extend(z[k])
                                b['offset'].extend((z['offset'][1:]+b['offset'][-1]).tolist())
                        b={k:np.asarray(x) for k,x in b.items()}
                    for s in final.SEEDS:
                        path=current(d,'val21',s,v)
                        with np.load(path) as p:a=pack(m,p,v+'_seed'+str(s),th)
                        contrast(a,b,f'{d}_{v}_seed{s}_minus_actual_native',th,dict(domain=d,role='val21',seed=s,variant=v,current=ev(path),native_review=ev(src/'terminal.json'),system_not_module=True),done)
            assert len(done)==390,len(done)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',decision=decision,panels=ev(OUT/'panels.json'),
            panel_count=len(done),independent_statistics=True,shared_accepted_matching=True,wall_s=time.monotonic()-begin))
        record('accepted_component_direct_contrasts_v1',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),panels=len(done)))
        record('accepted_component_direct_contrasts_v1',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
