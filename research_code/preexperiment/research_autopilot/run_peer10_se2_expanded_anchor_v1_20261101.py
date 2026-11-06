"""K2/K3 real cached AP tournament, causal ego-transported SE(2) state."""
from __future__ import annotations
import argparse, json, os, resource, time, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import run_motion_direction_information_state_fitcal_v1_20261031 as core
import run_peer07_spd_gt_source_split_ap_v1_20261031 as source
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,queues,sha

J=np.asarray([[0.,-1.],[1.,0.]])
def observations(m,i,expanded=False,score_floor=0.):
    a,b=m['condition_offsets'][i:i+2].astype(int);remote=a+np.flatnonzero(m['source'][a:b]==1)
    ego=a+np.flatnonzero((m['source'][a:b]==0)&((m['protected'][a:b]==1)|expanded))
    ego=ego[m['raw_scores'][ego]>=score_floor]
    # Candidate pool can contain repeated ego hypotheses: one measurement per geometry.
    if len(ego):
        _,unique=np.unique(np.round(m['cv_boxes'][ego,:7].astype(np.float64),4),axis=0,return_index=True);ego=ego[np.sort(unique)]
    remote_boxes,_=core.heading_projection(m['stale_boxes'][remote],m['cv_boxes'][remote]);ego_boxes=m['cv_boxes'][ego]
    if not len(remote) or not len(ego):return np.empty((0,2)),np.empty((0,2)),np.empty(0),np.empty((0,2),np.int64)
    ri,ei=core.pairs(remote_boxes,ego_boxes,2.5,True)
    age=m['cv_features'][remote[ri],0].astype(np.float64)/1000
    speed=np.linalg.norm(m['cv_boxes'][remote[ri],:2]-m['stale_boxes'][remote[ri],:2],axis=1)/np.maximum(age,.001)
    return remote_boxes[ri,:2].astype(np.float64),ego_boxes[ei,:2].astype(np.float64)-remote_boxes[ri,:2],speed,np.column_stack([remote[ri],ego[ei]])

def state(m,rows,prior=.3,pivot_mode='infra',expanded=False,score_floor=0.,static_theta=False,constant_theta=None,translation_only=False):
    count=len(rows);means=np.zeros((count,3));covs=np.zeros((count,3,3));pivots=np.zeros((count,2));counts=np.zeros(count,np.int16);expanded_counts=np.zeros(count,np.int16);states={};seen=set()
    for i,row in enumerate(rows):
        key=(str(m['sequence'][i]),int(m['wait_ms'][i]));stamp=int(m['receiver_timestamp_us'][i]);identity=(*key,stamp)
        if identity in seen:raise RuntimeError(('repeated observation',identity))
        seen.add(identity);pose=np.asarray(row['current_ego_pose_lidar_to_world'],np.float64)
        if key in states:
            old,P,oldpose,oldpivot,oldstamp=states[key];assert stamp>oldstamp
            relative=np.linalg.solve(pose,oldpose);R=relative[:2,:2];u=relative[:2,3]
        else:old=P=oldpose=oldpivot=None;R=np.eye(2);u=np.zeros(2)
        if pivot_mode=='ego':pivot=np.zeros(2)
        elif row.get('source') is not None:pivot=np.asarray(row['source']['source_to_current_ego'],np.float64)[:2,3]
        elif oldpivot is not None:pivot=R@oldpivot+u
        else:pivot=np.zeros(2)
        if old is None:
            mean=np.asarray([0.,0.,0. if constant_theta is None else constant_theta]);P=np.diag([1.,1.,np.deg2rad(prior)**2])
        else:
            A=np.eye(3);A[:2,:2]=R;A[:2,2]=J@(pivot-u-R@oldpivot)
            mean=A@old;P=A@P@A.T
        P=P+np.diag([.05**2,.05**2,np.deg2rad(.01)**2])
        points,residual,speed,pair_ids=observations(m,i)
        counts[i]=len(points)
        if expanded and not len(points):points,residual,speed,pair_ids=observations(m,i,True,score_floor)
        expanded_counts[i]=len(points)
        if translation_only or constant_theta is not None:
            theta=0. if translation_only else float(constant_theta);mean[2]=theta
            if len(points):
                z=residual-theta*(points-pivot)@J.T;precision=np.eye(2)/(.6**2/len(z));prior_info=np.linalg.inv(P[:2,:2]);newP=np.linalg.inv(prior_info+precision)
                mean[:2]=newP@(prior_info@mean[:2]+precision@z.mean(axis=0));P[:2,:2]=newP
            P[:2,2]=0;P[2,:2]=0
        elif len(points):
            H=np.zeros((len(points),2,3));H[:,:,:2]=np.eye(2);H[:,:,2]=(points-pivot)@J.T
            if static_theta:H[speed>.5,:,2]=0
            precision=np.linalg.inv(P);I=np.einsum('nji,njk->ik',H,H)/.6**2;rhs=np.einsum('nji,nj->i',H,residual)/.6**2
            P=np.linalg.inv(precision+I);mean=P@(precision@mean+rhs)
        P=(P+P.T)/2
        if not np.isfinite(mean).all() or not np.isfinite(P).all() or np.linalg.eigvalsh(P).min()<=0:raise RuntimeError('state finite/PSD')
        states[key]=(mean.copy(),P.copy(),pose.copy(),pivot.copy(),stamp);means[i]=mean;covs[i]=P;pivots[i]=pivot
    return {'mean':means,'covariance':covs,'pivot':pivots,'protected_pair_count':counts,'used_pair_count':expanded_counts}

def geometry(m,s):
    boxes,features=core.build_heading(m,np.zeros((len(m['wait_ms']),2)))
    matrices=np.tile(np.eye(4),(len(m['wait_ms']),1,1))
    for i,(value,pivot) in enumerate(zip(s['mean'],s['pivot'])):
        theta=value[2];R=np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);t=pivot+value[:2]-R@pivot
        matrices[i,:2,:2]=R;matrices[i,:2,3]=t
        a,b=m['condition_offsets'][i:i+2].astype(int);remote=a+np.flatnonzero(m['source'][a:b]==1);anchors=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1))
        boxes[remote,:2]=(boxes[remote,:2].astype(np.float64)@R.T+t).astype(np.float32);boxes[remote,6]=((boxes[remote,6]+theta+np.pi)%(2*np.pi)-np.pi).astype(np.float32)
        features[remote]=core.fixed.base.actual.optional.update_features(m['cv_features'][remote],boxes[remote],m['cv_boxes'][anchors],m['raw_scores'][anchors])
    assert np.array_equal(boxes[m['source']==0],m['cv_boxes'][m['source']==0])
    return boxes,features,matrices

def score(m,f,c):return core.refit.score_with_head(m,f,ROOT/Path(c['inputs']['matched_q005_head']['path']).parent)
def evaluate(m,states,c,baseline=None):
    boxes={};scores={};matrices={}
    for name,s in states.items():
        boxes[name],f,matrices[name]=geometry(m,s);scores[name]=score(m,f,c)
    if baseline is not None:
        boxes['K1_exact'],f=core.build_heading(m,baseline);scores['K1_exact']=score(m,f,c)
    panels,pred=core.fixed.base.evaluate(m,boxes,scores)
    return panels,pred,boxes,matrices

def extend_masks(codes,group):
    if group=='all':return np.ones(len(codes),bool),np.zeros(len(codes),bool)
    target=(codes!=2) if group=='veh_coop' else codes=={'veh':0,'coop':1,'inf':2}[group]
    return target,~target

def synthetic_checks():
    rng=np.random.RandomState(3);oldpivot=rng.randn(2);newpivot=rng.randn(2);phi=.3;R=np.asarray([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]);u=rng.randn(2);x=np.asarray([.2,-.4,.003]);A=np.eye(3);A[:2,:2]=R;A[:2,2]=J@(newpivot-u-R@oldpivot);xn=A@x;p=rng.randn(12,2)
    left=(x[:2]+x[2]*(p-oldpivot)@J.T)@R.T;pn=p@R.T+u;right=xn[:2]+xn[2]*(pn-newpivot)@J.T
    assert np.allclose(left,right,atol=1e-12)
    return {'linearized_field_ego_and_pivot_transport_max_error':float(np.max(np.abs(left-right))),'zero_theta_rigid_equals_translation':True,'state_observation':'first order, output geometry exact SE2','not_AP':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();c=load(a.contract)
    try:
        atomic(out/'synthetic_checks.json',synthetic_checks())
        for name,spec in c['inputs'].items():
            if name.endswith(('material','conditions','frame')) and sha(ROOT/spec['path'])!=spec['sha256']:raise RuntimeError(('input SHA drift',name))
        oldrows=core.load_jsonl(ROOT/c['inputs']['fit_conditions']['path']);trainrows=core.load_jsonl(ROOT/c['inputs']['train_conditions']['path']);valrows=core.load_jsonl(ROOT/c['inputs']['val_conditions']['path'])
        fit=core.load_npz(ROOT/c['inputs']['fit24_material']['path']);rows=core.conditions_for(fit,oldrows);fitdelta=core.load_npz(ROOT/c['inputs']['fit_q005_frame']['path'])['q005_delta']
        # Fit all hypotheses before seeing cal/hold/val; only three SE2 scales and two K3 variants.
        fitstates={f'prior_{v}':state(fit,rows,v) for v in c['K2']['theta_prior_degrees']}
        fp,_,_,_=evaluate(fit,fitstates,c,fitdelta)
        means={name:float(np.mean([r['ap70'] for r in fp if r['method']==name])) for name in fitstates}
        selected=max(means,key=lambda n:(means[n],-float(n.split('_')[1])));prior=float(selected.split('_')[1]);constant=float(np.median(fitstates[selected]['mean'][:,2]))
        k3states={name:state(fit,rows,prior,expanded=True,score_floor=floor,translation_only=True) for name,floor in [('expanded_all',0.),('expanded_score030',.3)]}
        kp,_,_,_=evaluate(fit,k3states,c,fitdelta);kmeans={name:float(np.mean([r['ap70'] for r in kp if r['method']==name])) for name in k3states};k3=max(kmeans,key=kmeans.get);floor=0. if k3=='expanded_all' else .3
        freeze={'status':'fit_selected_before_cal_and_formal','theta_prior_degrees':prior,'theta_fit_constant_rad':constant,'K3':k3,'K3_score_floor':floor,'fit_panels':fp+kp,'selection_scores':means,'K3_selection_scores':kmeans,'no_per_wait_or_formal_selection':True};atomic(out/'fit_freeze.json',freeze)
        # Actual same-environment numerical replay; AP tolerance is not a promotion threshold.
        replay=state(fit,rows,prior);assert np.array_equal(replay['mean'],fitstates[selected]['mean']);atomic(out/'light_replay.json',{'status':'passed','state_mean_exact':True,'state_covariance_exact':np.array_equal(replay['covariance'],fitstates[selected]['covariance'])})
        summaries=[];seqrows=[]
        for role in ('calibration8','val21','mechanism_holdout14'):
            key=role+'_material';m=core.load_npz(ROOT/c['inputs'][key]['path']);ledger=oldrows if role=='calibration8' else valrows if role=='val21' else trainrows;rows=core.conditions_for(m,ledger)
            states={'K2_infra':state(m,rows,prior),'K2_ego_pivot':state(m,rows,prior,pivot_mode='ego'),'K2_free_theta':state(m,rows,3.),'K2_static_theta':state(m,rows,prior,static_theta=True),'K2_fit_constant_theta':state(m,rows,prior,constant_theta=constant),'K3_K1_expanded':state(m,rows,prior,expanded=True,score_floor=floor,translation_only=True),'K3_K2_expanded':state(m,rows,prior,expanded=True,score_floor=floor)}
            if role=='calibration8':baseline=core.load_npz(ROOT/'artifacts/research_autopilot/status_audit_20261029_strict60_q005_task_refit_portability_v1/producer_v1/calibration8_frame_and_residual.npz')['q005_delta']
            else:baseline=core.load_npz(ROOT/c['inputs'][role+'_gate_material']['path'])['q005_base_delta']
            # Expanded translation equals exact K1 on frames which already have protected anchors;
            # only previously unsupported frames use the new causal state.
            expanded=states['K3_K1_expanded'];expanded['mean'][expanded['protected_pair_count']>0,:2]=baseline[expanded['protected_pair_count']>0]
            panels,pred,boxes,matrices=evaluate(m,states,c,baseline)
            if role=='val21':
                core.save_npz(out/'K2_val21_native_manifest.npz',sequence=m['sequence'],decision_us=m['receiver_timestamp_us'],wait_ms=m['wait_ms'],global_index=np.asarray([int(r['global_index']) for r in rows]),matrix=matrices['K2_infra'],remote_present=np.asarray([r.get('source') is not None for r in rows]))
                atomic(out/'native_manifest_ready.json',{'status':'ready','manifest':ev(out/'K2_val21_native_manifest.npz'),'cases':len(rows),'theta_prior_fit_frozen':prior,'GT_used_in_inference':False})
            core.save_npz(out/(role+'_state.npz'),**{name+'__'+key:value for name,s in states.items() for key,value in s.items()})
            core.save_npz(out/(role+'_predictions.npz'),**pred)
            if role!='calibration8':
                codes,mapping=source.map_provenance(m,ledger,role,ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label')
                assert mapping['unmatched']==mapping['ambiguous']==0
                source.METHODS=tuple(boxes);source.GROUPS=('all','veh','coop','inf','veh_coop');source.masks=extend_masks
                sp,sq,outcomes,exact=source.evaluate_role(m,pred,boxes,codes,role);assert all(exact.values());summaries.extend(sp);seqrows.extend(sq);core.save_npz(out/(role+'_source_outcomes.npz'),**outcomes);atomic(out/(role+'_mapping.json'),mapping)
            else:summaries.extend(dict(r,role=role,source_group='all') for r in panels)
            atomic(out/'panels.json',summaries);atomic(out/'per_sequence_panels.json',seqrows);atomic(out/'status.json',{'status':'running','completed_role':role,'wall_s':time.monotonic()-start})
        terminal={'status':'complete_light_checked','candidate_ids':['K1','K2','K3'],'contract':ev(a.contract),'fit_freeze':ev(out/'fit_freeze.json'),'panels':ev(out/'panels.json'),'per_sequence':ev(out/'per_sequence_panels.json'),'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'backbone_updates':0},'review_level':'exploratory formal + same environment replay; full independent review pending if top3','remaining':['K2 own geometry matched U','K2 Where/ViT actual inputs','K2/K3 V2V4Real current recipe and pre-registered fallback','auxiliary label protocol all methods','native K2 GPU']}
        atomic(out/'terminal.json',terminal);queues({k:{'status':'complete_light_checked' if k=='K1' else 'running_successor_chain','cached_formal_terminal':ev(out/'terminal.json'),'next_action':terminal['remaining']} for k in ('K1','K2','K3')})
    except Exception as e:
        atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'wall_s':time.monotonic()-start});raise
if __name__=='__main__':main()
