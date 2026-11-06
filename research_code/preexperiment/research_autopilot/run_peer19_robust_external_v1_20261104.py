"""R4/E3b: cal-selected robust external translations, exact same H1 readout.

Median over last K message slots (only successful estimates contribute), or
KF dropping innovations beyond squared normalized gate. Preserve instantaneous
centered displacement field, tilt, yaw, dimensions and z; no external rerun.
Primary fair comparison uses the SAME selected H1 three heads and prefix tau
for H1 and every module. All modules/filters reported, global robust choice is
cal-only. Module-level evidence, never a full external-system reproduction.
"""
import argparse,time,traceback
from collections import deque
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_recipe_v3_20261104 as recipe
import run_peer16_E5_module_head_seeds_v1_20261103 as old
from run_peer16_E3_external_temporal_v2_20261103 import translations
OUT=BASE/'robust_external'
REC=BASE/'recipe_v3_interface_recovery_v2'
METHODS=('freealign','roco','cbm')
GRIDS=dict(median=(3,5,10,20),gated_KF=(4.,9.,16.))

def filtered(m,z,valid,kind,param,q):
    states={};out=[];accepted=[]
    for seq,w,y,ok in zip(m['sequence'],m['wait_ms'],z,valid):
        key=str(seq),int(w);x,p,hist=states.get(key,(np.zeros(2),1.,deque()));x=x.copy();used=False
        if kind=='instant':x=y.copy();used=bool(ok)
        elif kind=='median':
            hist.append(y.copy() if ok else None)
            while len(hist)>param:hist.popleft()
            ys=[v for v in hist if v is not None]
            x=np.median(ys,axis=0) if ys else np.zeros(2);used=bool(ok)
        else:
            p+=q*q
            if ok and float(np.sum((y-x)**2)/(p+.36))<=param:
                k=p/(p+.36);x+=k*(y-x);p*=1-k;used=True
        states[key]=(x.copy(),p,hist);out.append(x.copy());accepted.append(used)
    return np.asarray(out),np.asarray(accepted)

def material(role,method,data,spec):
    if role=='calibration8':
        m,rows=data[role]
        path=old.OUT/role/(role+('_freealign_materialized.npz' if method=='freealign' else '_module_material.npz'))
    else:
        m,rows,_,_,_=old.modules(role,spec);path=ROOT/spec['inputs'][role+('_freealign' if method=='freealign' else '_module')]['path']
    with np.load(path) as z:
        b=z[method+'_boxes'].copy();f=z[method+'_features'].copy()
        valid=(z['status_code']==list(z['status_names']).index('estimated')) if method=='freealign' else (z[method+'_status']=='estimated')
    target,_=data[role]
    keys=('source','raw_scores','condition_offsets','sequence','wait_ms','receiver_timestamp_us','gt_offsets','gt_boxes','cv_boxes','stale_boxes')
    same={k:np.array_equal(m[k],target[k]) for k in keys};assert all(same.values()),(role,method,same)
    return m,rows,b,f,valid,path

def geometry(m,b,f,z,d):
    bb=b.copy();ff=f.copy()
    for i,(lo,hi) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
        lo,hi=int(lo),int(hi);ids=lo+np.flatnonzero(m['source'][lo:hi]==1);fixed=lo+np.flatnonzero(m['protected'][lo:hi]==1)
        bb[ids,:2]+=(d[i]-z[i]).astype(np.float32)
        ff[ids]=old.author.formal.update_features(f[ids],bb[ids],bb[fixed],m['raw_scores'][fixed])
    ff[:,recipe.prior.REMOVED]=0
    return bb,ff

def prepare():
    register('robust_external',__file__,__doc__,dict(GPU_forwards=0,module_calls=0,updates=0,CPU_threads=1,
        wall_h='bounded30h',disk_MiB='<200 compact scores/states; GT and module cache referenced'),
        methods=list(METHODS),grids={k:list(v) for k,v in GRIDS.items()},
        producer=ev(BASE/'recipe_global_choice_interface_recovery_v2/contract.json'),
        cached_modules=ev(old.OUT/'terminal.json'),module_contract=ev(old.OLD),
        q='H1 author cal-selected q',tau='H1 author cal-selected tau',
        selection='Mean cal8 AP70 across3heads/all4waits; choose parameter per method/filter, then global method/filter. Tie lexicographic method/filter and lower parameter.',
        preregistered_read='Global cal-selected robust external val21 mean>=H1-.3pp weakens claim to observation source can change; never select by formal cells.')
    register('robust_external_independent',__file__,'Independent polygons/AP/TP and paired whole-sequence2000CI for every instant/robust external vs exact same H1 three heads, both formal roles. Not independent detector/module fitting.',dict(GPU_forwards=0,CPU_threads=1,replicates=2000,disk_MiB='<50 draws/checks'),producer=ev(OUT/'contract.json'))

def run():
    begin=time.monotonic();record('robust_external',dict(status='waiting_H1_calibration_only_global_readout_choice',resource_owner=False,owner=None))
    try:
        wait(BASE/'recipe_global_choice_interface_recovery_v2/terminal.json',30)
        registry=load(REC/'method_registry.json');kind=registry['selected_kind'];cal=load(REC/'author/calibration.json');q=cal['q'];tau=cal['choices'][kind]['tau']
        recipe.OUT=REC;heads={s:recipe.head('author',kind,s) for s in recipe.SEEDS}
        data,_,_=recipe.materials('author');_,cs=old.configs();spec=cs['modules'];choices={};calpanels=[];formal=[];identities=[]
        record('robust_external',dict(status='running_cal8_robust_parameter_selection',resource_owner=True))
        for method in METHODS:
            m,rows,b,f,valid,path=material('calibration8',method,data,spec);z,err=translations(m,b,valid)
            assert err<1e-10
            for filter,grid in GRIDS.items():
                for param in grid:
                    d,used=filtered(m,z,valid,filter,param,q);bb,ff=geometry(m,b,f,z,d)
                    for seed,head in heads.items():
                        label=method+'_'+filter+'_'+str(param)+'_seed'+str(seed);pp,p=recipe.prior.old.author.refit.fixed.base.evaluate(recipe.prior.protected(m,tau),{label:bb},{label:recipe.score(m,ff,head)})
                        calpanels.extend(dict(r,module=method,filter=filter,param=param,seed=seed) for r in pp)
                    atomic(OUT/'calibration_panels.json',calpanels)
                values=[dict(param=p,AP70=float(np.mean([r['ap70'] for r in calpanels if r['module']==method and r['filter']==filter and r['param']==p]))) for p in grid]
                choices[method+'_'+filter]=max(values,key=lambda r:(r['AP70'],-r['param']))
        selected=sorted(choices,key=lambda k:(-choices[k]['AP70'],k))[0]
        atomic(OUT/'calibration.json',dict(created_at=now(),selected_global=selected,choices=choices,kind=kind,q=q,tau=tau,
            heads={str(s):ev(h/'direct_u.pth') for s,h in heads.items()},selection_roles=['calibration8'],no_formal_selection=True))
        for role in ('val21','mechanism_holdout14'):
            for method in METHODS:
                m,rows,b,f,valid,path=material(role,method,data,spec);z,err=translations(m,b,valid)
                identities.append(dict(role=role,module=method,input=ev(path),H1_population_geometry_exact=True,valid_messages=int(valid.sum()),centroid_decomposition_error=err))
                for filter in ('instant',*GRIDS):
                    param=None if filter=='instant' else choices[method+'_'+filter]['param'];d,used=filtered(m,z,valid,filter,param,q);bb,ff=geometry(m,b,f,z,d)
                    tag=method+'_'+filter;np.savez_compressed(OUT/(role+'_'+tag+'_state.npz'),observation=z,valid=valid,delta=d,accepted=used)
                    for seed,head in heads.items():
                        label=tag+'_seed'+str(seed);pp,p=recipe.prior.old.author.refit.fixed.base.evaluate(recipe.prior.protected(m,tau),{label:bb},{label:recipe.score(m,ff,head)})
                        formal.extend(dict(r,role=role,module=method,filter=filter,param=param,seed=seed,q=q,tau=tau,H1_head_kind=kind) for r in pp)
                        np.savez_compressed(OUT/(role+'_'+label+'_predictions.npz'),**p);atomic(OUT/'panels.json',formal)
                        # Same environment replay before discussing any float differences.
                        if seed==recipe.SEEDS[0] and filter=='instant':
                            _,repeat=recipe.prior.old.author.refit.fixed.base.evaluate(recipe.prior.protected(m,tau),{label:bb},{label:recipe.score(m,ff,head)})
                            assert all(np.array_equal(v,repeat[k]) for k,v in p.items())
                    atomic(OUT/'progress.json',dict(created_at=now(),role=role,module=method,filter=filter,panels=len(formal)))
        h=load(REC/'author/panels.json');hv=float(np.mean([r['ap70'] for r in h if r['role']=='val21' and r['kind']==kind]));evv=float(np.mean([r['ap70'] for r in formal if r['role']=='val21' and r['module']+'_'+r['filter']==selected]))
        atomic(OUT/'preregistered_read.json',dict(created_at=now(),cal_selected_external=selected,H1_AP70=hv,external_AP70=evv,external_minus_H1=evv-hv,
            weaken_claim=bool(evv>=hv-.3),formal_role='val21',threshold_pp=-.3,independent_pending=True,all_other_formal_arms_reported=True))
        atomic(OUT/'identity.json',identities)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(OUT/'panels.json'),read=ev(OUT/'preregistered_read.json'),wall_s=time.monotonic()-begin,independent_pending=True))
        record('robust_external',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('robust_external',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

def review():
    from review_peer16_E4_E7_AP_v1_20261103 import check
    from review_peer19_recipe_v3_20261104 import contrast
    dest=BASE/'robust_external_independent';results=[];boots=[]
    try:
        wait(OUT/'terminal.json',32);data,_,_=recipe.materials('author');_,cs=old.configs();spec=cs['modules'];cal=load(OUT/'calibration.json');gold=load(OUT/'panels.json')
        for role in ('val21','mechanism_holdout14'):
            for method in METHODS:
                m,rows,b,f,valid,path=material(role,method,data,spec)
                for filter in ('instant',*GRIDS):
                    tag=method+'_'+filter;bb=b.copy()
                    with np.load(OUT/(role+'_'+tag+'_state.npz')) as z:
                        for i,(lo,hi) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
                            lo,hi=int(lo),int(hi);ids=np.where(m['source'][lo:hi]==1)[0]+lo;bb[ids,:2]+=(z['delta'][i]-z['observation'][i]).astype(np.float32)
                    for seed in recipe.SEEDS:
                        label=tag+'_seed'+str(seed);path=OUT/(role+'_'+label+'_predictions.npz')
                        with np.load(path) as p:
                            rr=check(m,bb,p,label,[r for r in gold if r['role']==role]);assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70));results.extend(dict(r,role=role,module=method,filter=filter,seed=seed) for r in rr)
                            with np.load(REC/'author'/(role+'_'+cal['kind']+'_seed'+str(seed)+'_predictions.npz')) as h:
                                contrast(m,p,label,h,'H1_'+cal['kind']+'_seed'+str(seed),role+'_'+label+'_minus_H1',dict(role=role,module=method,filter=filter,seed=seed),dest,boots)
                        atomic(dest/'AP.json',results);atomic(dest/'bootstrap.json',boots)
        atomic(dest/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),AP=ev(dest/'AP.json'),bootstrap=ev(dest/'bootstrap.json'),same_H1_weights_prefix=True,independent_training=False))
        record('robust_external_independent',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('robust_external_independent',dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','review']);globals()[p.parse_args().mode]()
