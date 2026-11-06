"""Temporalize original external-module translations, keeping their instant rotations.

This is the old strict60 module population, not the author checkpoint or full systems.
"""
import argparse
import time
import traceback
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
from bootstrap_peer15_paper_v1_20261103 import configs,contrast,from_pred
from run_peer10_aux_remaining_v2_20261101 import modules
from run_peer10_fixed_population_aux_v1_20261101 import verified,core
import run_peer15_author_recipe_v1_20261103 as author
OUT=BASE/'E3_external_temporal'

def prepare():
    _,cs=configs();c=cs['modules']
    register('E3_external_temporal',__file__,
        'External per-message translations plus causal KF/EMA may explain part of CoMET gain. Retain instant rotation and every positive/negative result; never present these old strict60 module identities as the current author method or complete external systems.',
        dict(GPU_forwards=0,CPU_threads=1,updates=0,wall_h='1-6 plus paired bootstrap',disk_MiB='<100 compact predictions/states; no copied pools'),
        inputs=c['inputs'],head=c['input_weight_identity']['head'],
        filter=dict(q_m=.05,r_m=.6,EMA_alpha=.25,selection='Fixed transfer controls, no val/hold fitting; one module estimate per successful message, no invented anchor count'),
        methods=['freealign','roco','cbm'],roles=['mechanism_holdout14','val21'],
        geometry='Relative SE2 recovered from saved stale/module boxes; exact instant yaw retained. Smooth translation only; no new heading/CV motion, detector, association, training or method calls.',
        boundary=c['boundary'],statistics='2000 paired whole-sequence resamples, both readouts/AP thresholds/all waits; independent review remains separate')

def translations(m,b,valid):
    z=[];maxerr=0.
    for i,(a,e) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
        a,e=int(a),int(e);ids=a+np.flatnonzero(m['source'][a:e]==1)
        if not len(ids):z.append(np.zeros(2));continue
        before=m['stale_boxes'][ids];after=b[ids]
        angle=(after[:,6].astype(float)-before[:,6]+np.pi)%(2*np.pi)-np.pi
        # A single rigid transform; use circular mean across wrapped box headings.
        theta=np.arctan2(np.sin(angle).mean(),np.cos(angle).mean())
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        shift=after[:,:2].astype(float)-before[:,:2]@R.T;t=shift.mean(0)
        err=float(np.max(abs(shift-t),initial=0));maxerr=max(maxerr,err)
        assert err<1e-3,('nonrigid_cached_module',i,err)
        z.append(t)
    return np.asarray(z),maxerr

def filtered(m,z,valid,kind):
    state={};out=[]
    for seq,w,y,ok in zip(m['sequence'],m['wait_ms'],z,valid):
        key=str(seq),int(w);x,p=state.get(key,(np.zeros(2),1.));x=x.copy();p+=.05**2
        if ok:
            k=p/(p+.36) if kind=='KF' else .25;x+=k*(y-x)
            if kind=='KF':p*=1-k
        state[key]=(x,p);out.append(x.copy())
    return np.asarray(out)

def run():
    start=time.monotonic();cpu=time.process_time();record('E3_external_temporal',dict(status='running_cached_module_temporal_controls'))
    try:
        _,cs=configs();c=cs['modules'];head=ROOT/c['input_weight_identity']['head']['path'];head=head.parent
        for role in ('mechanism_holdout14','val21'):
            m,rows,old,boxes,_=modules(role,c);fa=core.load_npz(verified(c['inputs'][role+'_freealign']));mod=core.load_npz(verified(c['inputs'][role+'_module']))
            panels=[];pred={};states={};checks=[]
            for method in ('freealign','roco','cbm'):
                b=boxes[method+'__raw'];f=fa['freealign_features'] if method=='freealign' else mod[method+'_features']
                valid=(fa['status_code']==list(fa['status_names']).index('estimated')) if method=='freealign' else (mod[method+'_status']=='estimated')
                z,err=translations(m,b,valid);states[method+'__observation']=z;states[method+'__valid']=valid
                for arm in ('instant','KF','EMA'):
                    d=z if arm=='instant' else filtered(m,z,valid,arm);bb=b.copy();ff=f.copy()
                    if arm!='instant':
                        for i,(a,e) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
                            a,e=int(a),int(e);ids=a+np.flatnonzero(m['source'][a:e]==1);fixed=a+np.flatnonzero(m['protected'][a:e]==1)
                            bb[ids,:2]+=(d[i]-z[i]).astype(np.float32)
                            ff[ids]=author.formal.update_features(f[ids],bb[ids],bb[fixed],m['raw_scores'][fixed])
                    name=method+'_'+arm;score=author.refit.score_with_head(m,ff,head)
                    pp,p=author.refit.fixed.base.evaluate(m,{name:bb,name+'_raw':bb},{name:score,name+'_raw':m['raw_scores']})
                    if arm=='instant':
                        for suffix,original in [('',method+'__fresh39d'),('_raw',method+'__raw')]:
                            for field in ('candidate','tp50','tp70','offset'):
                                assert np.array_equal(p[name+suffix+'__'+field],old[original+'__'+field]),(role,name,field)
                        checks.append(dict(method=method,instant_candidate_TP_offsets_exact=True,rigid_xy_max_error=err,valid_messages=int(valid.sum())))
                    panels.extend(pp);pred.update(p);states[name]=d
                    atomic(OUT/(role+'_panels.json'),panels);atomic(OUT/'status.json',dict(created_at=now(),role=role,method=name,stage='scoring',wall_s=time.monotonic()-start))
                del b,f
            pred.update(condition_sequence=m['sequence'],condition_wait_ms=m['wait_ms'])
            np.savez_compressed(OUT/(role+'_predictions.npz'),**pred);np.savez_compressed(OUT/(role+'_states.npz'),**states);atomic(OUT/(role+'_checks.json'),checks)
            results=[]
            for method in ('freealign','roco','cbm'):
                for arm in ('KF','EMA'):
                    for suffix in ('','_raw'):
                        for th in (50,70):
                            name=method+'_'+arm+suffix;baseline=method+'_instant'+suffix
                            results.extend(contrast(OUT,role+'_'+name+'_minus_instant',from_pred(m,pred,name,th),from_pred(m,pred,baseline,th),th,dict(role=role,head=ev(head/'direct_u.pth'),scope='old strict60 original modules, shared readout')))
                            atomic(OUT/(role+'_bootstrap.json'),results)
            del m,rows,old,boxes,fa,mod,pred
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,independent_pending=True))
        record('E3_external_temporal',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('E3_external_temporal',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
