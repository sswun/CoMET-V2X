"""Frozen-author-input CoMET integration, cal-only q and analysis-only sweeps."""
import argparse, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from peer15_v1_20261103 import ROOT,BASE,AUTO,atomic,ev,load,now,register,record
import run_shared_offset_optional_formal_v1_20261026 as formal
import run_crosscheckpoint_temporal_q005_task_refit_v1_20261029 as refit
import run_crosscheckpoint_current_q005_heading_geometry_peer06_v1_20261030 as heading
import run_peer10_fixed_population_aux_v1_20261101 as aux
from audit_common_offset_peer_v1_20261026 import pairs
from object_history_selector_core_v1_20261021 import PackedDataset,common_mask

OUT=BASE/'author_recipe'
CONTRACT=AUTO/'status_audit_20261031_peer07_geometry_matched_source_ap_v1/contract_v1.json'
Q=(.025,.05,.1,.15)

def prepare():
    c=load(CONTRACT);h=c['input_weight_identity']['geometry_heads']['kf'];head=ROOT/h['weights']['path']
    register('author_recipe',Path(__file__).resolve(),'Current heading+cal8-q may improve or degrade author q.15 CV recipe; preserve both and every one-factor sensitivity. No val/hold selection. Fixed author matched-U is not strict60 head transfer or a newly refitted U.',dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='2-8 including cal and full val one-factor sweeps',disk_MiB='<150 prediction ID/score deltas and primary geometry only; shared pools/GT/head referenced'),author_geometry_contract=ev(CONTRACT),head={k:ev(ROOT/v['path']) for k,v in h.items()},q_calibration_grid=list(Q),r_grid=[.3,.6,1.2],gate_grid=[1.5,2.5,3.5],anchor_score_grid='frozen protected threshold +/-0.1, clipped to[0,1]; estimator association only, readout protected prefix unchanged',selection='calibration8 pooled-AP70 four-wait mean, tie lower q; full val sensitivity is never selected',semantics='CV-anchor measurement before heading projection; scalar per-condition q variance, sequence/wait reset, no ego transport, matching current formal scalar branch; not new micro-mechanism')

def materials(role):
    c=load(CONTRACT)
    if role=='calibration8':
        d,_=formal.checked_dataset(ROOT/c['datasets']['mechanism_holdout14'])
        m=formal.materialize(d,role,np.zeros(2),np.zeros(2));gts=[d.pool(int(i))['gt_boxes'] for i in m['dataset_index']]
        m['gt_boxes']=np.concatenate(gts);m['gt_offsets']=np.r_[0,np.cumsum([len(g) for g in gts])];rows=[d.meta[int(i)] for i in m['dataset_index']]
        return m,rows,None
    g={name:formal.load_npz(aux.verified(c['inputs'][role+'_'+name])) for name in ('stale','cv','kf')}
    d=PackedDataset(ROOT/c['datasets'][role]);m=dict(g['cv']);m.update(cv_boxes=g['cv']['boxes'],cv_features=g['cv']['features'],stale_boxes=g['stale']['boxes'],stale_features=g['stale']['features'],kf_boxes=g['kf']['boxes'],kf_features=g['kf']['features'])
    m['gt_boxes']=np.concatenate([d.pool(int(i))['gt_boxes'] for i in m['dataset_index']]);rows=[d.meta[int(i)] for i in m['dataset_index']]
    pred=formal.load_npz(aux.verified(c['inputs'][role+'_predictions']))
    return m,rows,pred

def observations(m,gate=2.5,threshold=None):
    counts=[];measures=[];clock=time.process_time()
    for a,b in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        a,b=int(a),int(b);remote=a+np.flatnonzero(m['source'][a:b]==1)
        keep=(m['source'][a:b]==0)&((m['protected'][a:b]==1) if threshold is None else (m['raw_scores'][a:b]>=threshold))
        anchor=a+np.flatnonzero(keep);ri,ai=pairs(m['cv_boxes'][remote],m['cv_boxes'][anchor],gate,True)
        res=m['cv_boxes'][anchor[ai],:2].astype(np.float64)-m['cv_boxes'][remote[ri],:2]
        counts.append(len(ri));measures.append(res.mean(0) if len(res) else np.zeros(2))
    return np.asarray(counts),np.asarray(measures),time.process_time()-clock

def kf(m,n,z,q,r):
    state={};delta=[];variance=[];age=[];clock=time.process_time()
    for seq,wait,count,measurement in zip(m['sequence'],m['wait_ms'],n,z):
        key=str(seq),int(wait);mean,var,step=state.get(key,(np.zeros(2),1.,0));mean=mean.copy();prior=var+q*q
        if count:
            gain=prior/(prior+r*r/count);mean+=gain*(measurement-mean);var=(1-gain)*prior
        else:var=prior
        state[key]=(mean,var,step+1);delta.append(mean.copy());variance.append(var);age.append(step)
    return np.asarray(delta),np.asarray(variance),np.asarray(age),time.process_time()-clock

def evaluate(m,delta,head,name,use_heading=True):
    clock=time.process_time();boxes,features=heading.build_heading(m,delta) if use_heading else refit.task_geometry(m,delta);geometry_s=time.process_time()-clock
    clock=time.process_time();score=refit.score_with_head(m,features,head);readout_s=time.process_time()-clock
    panels,pred=refit.fixed.base.evaluate(m,{name:boxes},{name:score})
    return panels,pred,boxes,dict(geometry_and_feature_CPU_s=geometry_s,readout_CPU_s=readout_s)

def qualitative(m,rows,legacy,current,delta,n):
    # Prespecified first/middle/last of first two complete val sequences, wait400.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from review_traf_v2xseq_adapted_fixed60_raw_20261006 import polygons_hwl
    examples=[];seqs=sorted(set(map(str,m['sequence'])))[:2]
    for seq in seqs:
        ids=np.flatnonzero((m['sequence']==seq)&(m['wait_ms']==400))
        for i in ids[[0,len(ids)//2,-1]]:
            a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2]);ego=m['source'][a:b]==0
            p=OUT/'qualitative'/('frame_'+str(int(i))+'.npz');p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,gt=m['gt_boxes'][ga:gb],ego=m['cv_boxes'][a:b][ego],remote_before=m['cv_boxes'][a:b][~ego],remote_legacy=legacy[a:b][~ego],remote_current=current[a:b][~ego],delta=delta[i])
            examples.append(dict(condition=int(i),metadata=rows[i],anchors=int(n[i]),data=ev(p),selection='first/middle/last, first two sequences, wait400; no outcome selection'))
            fig,axs=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
            for ax,remote,title in zip(axs,[m['cv_boxes'][a:b][~ego],current[a:b][~ego]],['Before','CoMET']):
                for boxes,color in [(m['gt_boxes'][ga:gb],'black'),(m['cv_boxes'][a:b][ego],'tab:blue'),(remote,'tab:orange')]:
                    for polygon in polygons_hwl(boxes):ax.add_patch(Polygon(np.asarray(polygon.exterior.coords),fill=False,edgecolor=color,linewidth=.65))
                ax.set(xlim=(0,96),ylim=(-38.4,38.4),aspect='equal',title=title,xlabel='x (m)');ax.grid(alpha=.15)
            fig.suptitle(seq+' frame '+str(rows[i].get('current_vehicle_frame',i))+' | GT black, ego blue, remote orange')
            fig.tight_layout();fig.savefig(p.with_suffix('.pdf'));fig.savefig(p.with_suffix('.png'),dpi=150);plt.close(fig)
    atomic(OUT/'qualitative/examples.json',examples)

def run():
    start=time.monotonic();cpu=time.process_time();c=load(OUT/'contract.json');assert c['script']==ev(__file__)
    h=ROOT/c['head']['weights']['path'];head=h.parent
    for ref in c['head'].values():assert ev(ROOT/ref['path'])==ref
    record('author_recipe',dict(status='running_cal_only_recipe_integration'))
    try:
        m,rows,_=materials('calibration8');n,z,assoc=observations(m);cal=[]
        for q in Q:
            delta,_,_,_=kf(m,n,z,q,.6);pp,_,_,_=evaluate(m,delta,head,'heading_q'+str(q));cal.append(dict(q=q,mean_AP70=float(np.mean([r['ap70'] for r in pp])),panels=pp))
            atomic(OUT/'calibration_progress.json',cal)
        chosen=max(cal,key=lambda x:(x['mean_AP70'],-x['q']))['q']
        atomic(OUT/'calibration.json',dict(created_at=now(),selected_q=chosen,tables=cal,criterion=c['selection'],role='calibration8',before_val=True))
        m,rows,old=materials('val21');n,z,assoc=observations(m)
        # Qualify v1 state and geometry against accepted author cache before new reads.
        olddelta,_,_,_=kf(m,n,z,.15,.6);legacy,lf=refit.task_geometry(m,olddelta)
        assert np.array_equal(legacy,m['kf_boxes']),'legacy author geometry replay drift'
        legacy_score=refit.score_with_head(m,lf,head)
        pp,pred=refit.fixed.base.evaluate(m,{'v1_CV_q015_matchedU':legacy},{'v1_CV_q015_matchedU':legacy_score})
        checks={}
        for field in ('candidate','offset','tp50','tp70','score'):
            a=pred['v1_CV_q015_matchedU__'+field];b=old['kf__matched_head__'+field]
            checks[field]=dict(shape_equal=a.shape==b.shape,max_abs_error=float(np.max(np.abs(a-b),initial=0)) if a.shape==b.shape else None)
        atomic(OUT/'legacy_replay.json',dict(created_at=now(),geometry_exact=True,fields=checks,head=ev(h),same_environment=True))
        assert all(v['shape_equal'] and v['max_abs_error']<=1e-10 for v in checks.values()),'legacy prediction replay drift'
        delta,var,age,kfs=kf(m,n,z,chosen,.6);current,cp,boxes,timing=evaluate(m,delta,head,'current_heading_calq_matchedU')
        panels=pp+current;pred.update(cp);pred.update(condition_sequence=m['sequence'],condition_wait_ms=m['wait_ms'],condition_GT_count=np.asarray([int(common_mask(m['gt_boxes'][int(a):int(b)]).sum()) for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]))
        np.savez_compressed(OUT/'primary_predictions.npz',**pred)
        np.savez_compressed(OUT/'primary_geometry.npz',current_boxes=boxes,delta=delta,variance=var,anchors=n,observation=z,cold_step=age,condition_offsets=m['condition_offsets'])
        atomic(OUT/'primary_panels.json',panels);atomic(OUT/'input_rows.json',rows)
        timing.update(association_CPU_s=assoc,KF_CPU_s=kfs,conditions=len(n),association_ms_per_condition=1000*assoc/len(n),KF_ms_per_condition=1000*kfs/len(n),head=ev(h),scope='cached box module CPU timing, excludes detectors/transfer; not full system cost')
        atomic(OUT/'module_timing.json',timing);qualitative(m,rows,legacy,boxes,delta,n)
        # All one-dimensional variants retained, no best-row extraction.
        threshold=float(load(ROOT/load(CONTRACT)['datasets']['mechanism_holdout14']/'summary.json')['protect_threshold'])
        grid=[dict(name='q_'+str(q),q=q,r=.6,gate=2.5,threshold=None) for q in Q if q!=chosen]
        grid += [dict(name='r_'+str(r),q=chosen,r=r,gate=2.5,threshold=None) for r in (.3,1.2)]
        grid += [dict(name='gate_'+str(g),q=chosen,r=.6,gate=g,threshold=None) for g in (1.5,3.5)]
        grid += [dict(name='anchor_'+str(t),q=chosen,r=.6,gate=2.5,threshold=t) for t in sorted(set([max(0.,threshold-.1),min(1.,threshold+.1)]))]
        sweep=[]
        for spec in grid:
            nn,zz,_=observations(m,spec['gate'],spec['threshold']);dd,_,_,_=kf(m,nn,zz,spec['q'],spec['r']);p,_,_,_=evaluate(m,dd,head,spec['name']);sweep.append(dict(parameters=spec,panels=p));atomic(OUT/'sensitivity_panels.json',sweep)
            atomic(OUT/'status.json',dict(created_at=now(),status='analysis_only_fixed_readout_sensitivity',completed=len(sweep),planned=len(grid),wall_s=time.monotonic()-start))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',selected_q=chosen,primary_panels=ev(OUT/'primary_panels.json'),sensitivity=ev(OUT/'sensitivity_panels.json'),legacy_replay=ev(OUT/'legacy_replay.json'),independent_review_pending=True,no_val_selection=True,GPU_forwards=0,updates=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,no_model_wakeup=True))
        record('author_recipe',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('author_recipe',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();globals()[a.mode]()
