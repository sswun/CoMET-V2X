"""R6b five-domain three-head official AP and label-population2000CI.

Same H1 heads/q/prefix for geometry controls; original N0 identity separately
stated. R6a labels define post-hoc columns only. No population enters inference,
calibration selection or score computation; no hold-off variant is searched.
"""
import argparse,time,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import run_peer16_cached_methods_v1_20261103 as temporal
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry

METHODS=('H1','PUBLIC','single_mean','single_median','hold_last','EMA','window','OFFICIAL','OFFICIAL_filled','MA5','MA5_filled','N0','EGO')

def prepare():
    for domain in h1.recipe.DOMAINS:
        for review in (False,True):
            register('R6b_'+domain+('_independent' if review else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
                replicates=2000,disk_MiB='<220 per domain compact predictions/states/stats',wall_h='finite full analysis; no intermediate deadline'),
                domain=domain,methods=list(METHODS),populations=list(POPS),roles=list(ROLES),
                R6a=ev(BASE/'R6a_raw_labels_independent_v2/terminal.json'),registry=ev(REC/'method_registry.json'),
                coverage='Core geometry controls all5 domains/three heads/three roles/all4 waits. Native/official-late/external modules are separate R6b_external task, not silently claimed covered.',
                independence='Independent polygons/TP then independently implemented AP/bootstrap; same scored predictions/GT/training upstream')

def temporal_choice(m,q,tau,heads,domain):
    if domain=='author':return load(P19/'H1_assets/temporal/calibration.json')['selected']
    observations=h1.obs(m);choices={};panels=[]
    for method,grid in [('EMA',(.1,.25,.5,.75,1.)),('window',(3,5,10,20))]:
        for param in grid:
            d=temporal.temporal(m,observations,method,q,param);b,f=h1.recipe.geometry(m,d)
            for seed,path in heads.items():
                pp,_=h1.recipe.evaluate(m,b,f,path,'cal_'+method,tau)
                panels.extend(dict(r,method_name=method,param=param,seed=seed) for r in pp)
        choices[method]=max(grid,key=lambda p:(np.mean([r['ap70'] for r in panels if r['method_name']==method and r['param']==p]),-p))
    return choices

def run(domain,review=False):
    name='R6b_'+domain+('_independent' if review else '');dest=BASE/name;src=BASE/('R6b_'+domain);begin=time.monotonic()
    try:
        if review:wait(src/'terminal.json')
        wait(BASE/'R6a_raw_labels_independent_v2/terminal.json')
        record(name,dict(status='running_independent_populations' if review else 'running_core_geometry_population_AP_CI',resource_owner=True))
        registry=load(REC/'method_registry.json');kind=registry['selected_kind'];cal=load(REC/domain/'calibration.json');q=cal['q'];tau=cal['choices'][kind]['tau']
        h1.recipe.OUT=REC;heads={s:h1.recipe.head(domain,kind,s) for s in h1.recipe.SEEDS}
        dataset,_,oldq=data(domain);progress=[]
        if not review:
            choices=temporal_choice(dataset['calibration8'][0],q,tau,heads,domain)
            atomic(dest/'calibration.json',dict(selected=choices,q=q,tau=tau,kind=kind,selection_roles=['calibration8'],heads={str(s):ev(p/'direct_u.pth') for s,p in heads.items()}))
        else:choices=load(src/'calibration.json')['selected']
        for role in ROLES:
            m,rows=dataset[role];pops,off=annotations(rows);filled,filled_ma,span,supported=fill(rows,off)
            info={str(r['vehicle_frame']):r for r in load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json')}
            ma=reference_deltas(rows,info)['OFFICIAL_MA5'];observations=h1.obs(m)
            for method in METHODS:
                statepath=src/(role+'_'+method+'_state.npz')
                if review:
                    with np.load(statepath) as z:d=z['delta'].copy()
                else:
                    if method in ('PUBLIC','EGO'):d=np.zeros_like(off)
                    elif method=='H1':d=h1.recipe.hard_state(m,q)['delta']
                    elif method=='N0':d=temporal.temporal(m,temporal.observation(m),'KF',oldq)
                    elif method=='OFFICIAL':d=off
                    elif method=='OFFICIAL_filled':d=filled
                    elif method=='MA5':d=ma
                    elif method=='MA5_filled':d=filled_ma
                    else:d=temporal.temporal(m,observations,method,q,choices.get(method))
                    np.savez_compressed(statepath,delta=d)
                b,f=h1.recipe.geometry(m,d)
                if review:b=geometry(m,d)
                if method=='EGO':b[m['source']==1,:2]=1e6
                for seed,head in heads.items():
                    tag=role+'_'+method+'_seed'+str(seed);label=method+'_seed'+str(seed);path=src/(tag+'_predictions.npz')
                    if not review:
                        ht=load(h1.recipe.prior.OUT/domain/'calibration.json')['selected_tau'] if method=='N0' else tau
                        pp,p=h1.recipe.evaluate(m,b,f,head,label,ht)
                        atomic(src/(tag+'_AP.json'),pp);np.savez_compressed(path,**p)
                        save_stats(dest,m,rows,p,label,tag)
                    else:
                        with np.load(path) as z:p={k:z[k] for k in z.files}
                        rr=check(m,b,p,label,load(src/(tag+'_AP.json')))
                        assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70)),rr
                        save_stats(dest,m,rows,p,label,tag,True,load(src/(tag+'_statistics.json')))
                    progress.append(dict(role=role,method=method,seed=seed,prediction=ev(path)))
                    atomic(dest/'progress.json',dict(created_at=now(),completed=len(progress),last=tag))
        atomic(dest/'identity.json',dict(domain=domain,recipe=ev(REC/domain/'calibration.json'),current_registry=ev(REC/'method_registry.json'),
            temporal_calibration=ev(src/'calibration.json'),population=ev(BASE/'R6a_raw_labels/frames.json'),
            EGO='Same frozen matched readout ego-only diagnostic, not native ego-only system',N0='Original protected-anchor KF q and old no-position head/tau; not H1 diag renaming',
            label_assisted=['OFFICIAL','OFFICIAL_filled','MA5','MA5_filled'],head_seed_not_detector_seed=True,all_completed=progress))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
            completed=len(progress),wall_s=time.monotonic()-begin,external_native_coverage_separate=True,root_table_unchanged=True))
        record(name,dict(status='complete_independently_reviewed' if review else 'complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));p.add_argument('--domain',choices=h1.recipe.DOMAINS);a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.domain,a.mode=='review')
