"""Independent statistic successor: current author/native/stronger-late and modules.

Reuse independently accepted saved matches and current frozen recipes. All AP
integration/resampling arithmetic uses the separately written peer18 kernel,
never the production bootstrap functions. No forward, training or selection.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import review_peer18_final_native_statistics_v1_20261104 as independent
import run_peer17_final_recipe_v1_20261104 as final
import run_peer17_module_recipe_v1_20261104 as module
import bootstrap_peer17_main_contrasts_v1_20261104 as main
from object_history_selector_core_v1_20261021 import common_mask
OUT=BASE/'main_module_statistics_independent'

def pack(m,p,name,th):
    return dict(sequence=m['sequence'],wait=m['wait_ms'],
        gt=np.asarray([common_mask(m['gt_boxes'][int(a):int(b)]).sum() for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]),
        offset=p[name+'__offset'],score=p[name+'__score'],tp=p[name+'__tp'+str(th)])

def check(label,a,b,th,src,accepted,done):
    seqs=sorted(set(a['sequence'].astype(str)));assert seqs==sorted(set(b['sequence'].astype(str)))
    rng=np.random.Generator(np.random.PCG64(20261103));draw=rng.integers(0,len(seqs),(2000,len(seqs)))
    mult=np.asarray([np.bincount(r,minlength=len(seqs)) for r in draw],np.int16)
    with np.load(src/(label+f'_AP{th}.npz')) as z:
        assert np.array_equal(mult,z['cluster_counts']) and np.array_equal(seqs,z['sequence'].astype(str))
        expected={w:z['wait'+str(w)].copy() for w in (0,100,200,400)}
    dist=[];point=[];errors=[]
    for w in (0,100,200,400):
        x=independent.panel(a,w,seqs);y=independent.panel(b,w,seqs);assert np.array_equal(x[3],y[3])
        delta=100*(independent.empirical(*x,mult)-independent.empirical(*y,mult));dist.append(delta)
        pt=100*(independent.empirical(*x,np.ones((1,len(seqs)),np.int16))[0]-independent.empirical(*y,np.ones((1,len(seqs)),np.int16))[0]);point.append(float(pt))
        err=float(np.max(abs(delta-expected[w])));assert err<1e-8,(label,th,w,err);errors.append(err)
    for w,pt,dd in zip((0,100,200,400,'four_wait_mean'),point+[float(np.mean(point))],dist+[np.mean(dist,axis=0)]):
        gold=next(r for r in accepted if r['comparison']==label and r['threshold']==th and r['wait_ms']==w)
        ci=np.percentile(dd,[2.5,97.5]);err=max(abs(pt-gold['delta_pp']),abs(ci[0]-gold['CI95_low']),abs(ci[1]-gold['CI95_high']))
        assert err<1e-8,(label,th,w,err)
        done.append(dict(comparison=label,threshold=th,wait_ms=w,delta_pp=pt,CI95=ci.tolist(),maximum_draw_error=max(errors),maximum_summary_error=float(err),status='passed'))
    atomic(OUT/'panels.json',done);atomic(OUT/'progress.json',dict(created_at=now(),panels=len(done),comparison=label,threshold=th))

def prepare():
    register('main_module_statistics_independent',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        replicates=2000,panels=270,disk_MiB='<12',wall_h='finite24h'),
        predecessor=ev(independent.OUT/'contract.json'),kernel=ev(independent.__file__),
        current_main=ev(main.OUT/'terminal.json'),current_modules=ev(P17/'module_recipe_independent/terminal.json'),
        boundary='Independent statistics only; source matches/GT/recipe shared and independently checked upstream. Does not close all legacy paper statistics.')

def run():
    starttime=time.monotonic();record('main_module_statistics_independent',dict(status='waiting_four_domain_statistics',resource_owner=False,owner=None))
    done=[]
    try:
        while not (independent.OUT/'terminal.json').exists():
            assert time.monotonic()-starttime<24*3600;time.sleep(30)
        assert load(independent.OUT/'terminal.json')['status']=='complete_independently_reviewed'
        record('main_module_statistics_independent',dict(status='running_independent_main_module_statistics',resource_owner=True))
        data,_,_=final.old.materials('author');_,contracts=main.boot.configs();accepted=load(main.OUT/'panels.json')
        for role in ('val21','mechanism_holdout14'):
            m,mr=data[role];lm,lr,lp,_,_=main.boot.aux.official_late(role,contracts['official_late'])
            assert sorted((r['sequence'],r['wait_ms'],r['current_vehicle_frame']) for r in mr)==sorted((r['sequence'],r['wait_ms'],r['current_vehicle_frame']) for r in lr)
            for th in (50,70):
                seqs=sorted(set(lm['sequence'].astype(str)));candidates={n:pack(lm,lp,n,th) for n in ('official_late_no_comp','official_late_time_comp')}
                # Independently verify the frozen stronger column rule. Input
                # assembly retains original within-frame order and tie policy.
                combined=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[]);columns={}
                for w in (0,100,200,400):
                    aps={n:independent.empirical(*independent.panel(d,w,seqs),np.ones((1,len(seqs)),np.int16))[0] for n,d in candidates.items()}
                    name=max(aps,key=lambda n:(aps[n],n));columns[str(w)]=name;d=candidates[name]
                    for i in np.flatnonzero(d['wait']==w):
                        lo,hi=map(int,d['offset'][i:i+2]);combined['sequence'].append(d['sequence'][i]);combined['wait'].append(w);combined['gt'].append(d['gt'][i]);combined['score'].extend(d['score'][lo:hi]);combined['tp'].extend(d['tp'][lo:hi]);combined['offset'].append(len(combined['score']))
                combined={k:np.asarray(v) for k,v in combined.items()}
                native=None
                if role=='val21':
                    native=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[])
                    for w in (0,100,200,400):
                        with np.load(main.boot.BASE/'external_raw_traf'/f'wait{w:03d}_matches.npz') as z:
                            k='iou'+str(th);native['sequence'].extend(z['sequence']);native['wait'].extend([w]*len(z['sequence']));native['gt'].extend(z[k+'_frame_gt']);native['score'].extend(z[k+'_scores']);native['tp'].extend(z[k+'_tp']);native['offset'].extend((z[k+'_offset'][1:]+native['offset'][-1]).tolist())
                    native={k:np.asarray(v) for k,v in native.items()}
                for seed in final.SEEDS:
                    with np.load(final.OUT/'author'/f'{role}_seed{seed}_predictions.npz') as p:a=pack(m,p,'CoMET_seed'+str(seed),th)
                    label=f'{role}_CoMET_seed{seed}_minus_stronger_late'
                    assert next(r for r in accepted if r['comparison']==label and r['threshold']==th)['identity']['baseline_columns']==columns
                    check(label,a,combined,th,main.OUT,accepted,done)
                    if native is not None:check(f'val21_CoMET_seed{seed}_minus_same_author_native',a,native,th,main.OUT,accepted,done)
        _,cs=module.old.configs()
        for method in module.MODULES:
            src=P17/'module_recipe_independent'/method;accepted=load(src/'bootstrap.json')
            for role in ('val21','mechanism_holdout14'):
                m,_,_,_,_=module.old.modules(role,cs['modules'])
                for seed in final.SEEDS:
                    with np.load(module.OUT/method/f'seed{seed}'/(role+'_predictions.npz')) as p,np.load(final.OUT/'author'/f'{role}_seed{seed}_predictions.npz') as q:
                        for th in (50,70):
                            check(f'{role}_finalCoMET_minus_{method}_seed{seed}',pack(m,q,f'CoMET_seed{seed}',th),pack(m,p,f'{method}_U_seed{seed}',th),th,src,accepted,done)
                assert time.monotonic()-starttime<24*3600
        assert len(done)==270
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=ev(OUT/'panels.json'),
            independent_statistics_implementation=True,saved_matching_shared=True,whole_paper_statistics_closed=False,wall_s=time.monotonic()-starttime))
        record('main_module_statistics_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed_panels=len(done)))
        record('main_module_statistics_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
