"""New final3head author vs native and each column's stronger official late baseline.

Shared accepted2000 whole-sequence statistic; statistical independence is not
claimed. Cross-detector native gaps remain separate, never silently filled.
"""
import argparse,time,traceback
import numpy as np
from peer17_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
import bootstrap_peer15_paper_v1_20261103 as boot
OUT=BASE/'main_contrasts'
def prepare():
    register('main_contrasts',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<20 draws/metrics'),
        baseline=ev(final.OUT/'author/identity.json'),native=ev(boot.BASE/'external_raw_traf/terminal.json'),
        module_CI='Already complete in module_recipe_independent; reference rather than recalculate.')
def native(th):
    seq=[];waits=[];gt=[];score=[];tp=[];off=[0]
    for w in (0,100,200,400):
        with np.load(boot.BASE/'external_raw_traf'/('wait'+str(w).zfill(3)+'_matches.npz')) as z:
            k='iou'+str(th);seq.extend(z['sequence']);waits.extend([w]*len(z['sequence']));gt.extend(z[k+'_frame_gt'])
            score.extend(z[k+'_scores']);tp.extend(z[k+'_tp']);off.extend((z[k+'_offset'][1:]+off[-1]).tolist())
    return boot.make(seq,waits,gt,off,score,tp)
def stronger(m,p,th):
    parts=[];names={}
    for w in (0,100,200,400):
        candidates=[]
        for name in ('official_late_no_comp','official_late_time_comp'):
            d=boot.from_pred(m,p,name,th);x=boot.panel(d,w,sorted(set(d['sequence'])))
            candidates.append((boot.voc_ap(x[1],x[0],int(x[3].sum())),name,d))
        _,name,d=max(candidates,key=lambda r:(r[0],r[1]));names[str(w)]=name
        parts.extend((str(d['sequence'][i]),w,int(d['gt'][i]),d['score'][int(d['offset'][i]):int(d['offset'][i+1])],d['tp'][int(d['offset'][i]):int(d['offset'][i+1])]) for i in np.flatnonzero(d['wait']==w))
    off=np.r_[0,np.cumsum([len(r[3]) for r in parts])]
    return boot.make([r[0] for r in parts],[r[1] for r in parts],[r[2] for r in parts],off,np.concatenate([r[3] for r in parts]),np.concatenate([r[4] for r in parts])),names
def run():
    starttime=time.monotonic();rows=[]
    record('main_contrasts',dict(status='running_actual_new_recipe_contrasts',resource_owner=True))
    try:
        data,_,_=final.old.materials('author');_,contracts=boot.configs()
        for role in ('val21','mechanism_holdout14'):
            m,mr=data[role];lm,lr,lp,_,_=boot.aux.official_late(role,contracts['official_late'])
            key=lambda rr:sorted((str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])) for r in rr)
            assert key(mr)==key(lr)
            for th in (50,70):
                b,selected=stronger(lm,lp,th)
                for seed in final.SEEDS:
                    with np.load(final.OUT/'author'/(role+'_seed'+str(seed)+'_predictions.npz')) as p:a=boot.from_pred(m,p,'CoMET_seed'+str(seed),th)
                    identity=dict(role=role,seed=seed,recipe=ev(final.OUT/'author/identity.json'),baseline_columns=selected,
                        baseline_rule='Max existing official late per wait and threshold, per-column benchmark only, no CoMET selection.',shared_bootstrap_kernel=True)
                    rows.extend(boot.contrast(OUT,role+'_CoMET_seed'+str(seed)+'_minus_stronger_late',a,b,th,identity))
                    if role=='val21':rows.extend(boot.contrast(OUT,'val21_CoMET_seed'+str(seed)+'_minus_same_author_native',a,native(th),th,identity))
                    atomic(OUT/'panels.json',rows)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(OUT/'panels.json'),
            wall_s=time.monotonic()-starttime,independent_statistical_implementation=False,
            remaining='Cross-detector native contrasts and global independent statistics stay open; module CI already separately complete.'))
        record('main_contrasts',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
