"""Peer16 temporal baselines and joint clock-state experiments, cache-only."""
import argparse,json,time,traceback
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
from peer16_v1_20261103 import ROOT,AUTO,AUDIT,BASE,atomic,ev,load,now,register,record
import run_peer15_author_recipe_v1_20261103 as author
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
from run_peer16_E1_timing_audit_v1_20261103 import dynamics,match,fit
ROLES=('fit24','calibration8','mechanism_holdout14','val21')
DOMAINS=('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')
def cross_specs():
    s=load(AUTO/'status_audit_20261031_crosscheckpoint_motion_direction_information_state_current_inputs_v1/producer_v1/summary.json');ec=load(ROOT/s['contract']['path']);man=load(author.aux.verified(ec['inputs']['input_identity_manifest']));sc=load(author.aux.verified(man['source_contract']));return sc,man
def prepare():
    sc,man=cross_specs()
    register('E2_comet_delta',__file__,'Joint online [c_x,c_y,delta] may improve speed-varying/no-anchor conditions or absorb mismatches and harm AP. V0-V4 all retained; preregistered decision is not a general minimum-effect gate. No eval-cell choice.',dict(GPU_forwards=0,CPU_threads=1,wall_h='4-14 finite all5 detector domains, both roles/readouts',disk_MiB='<250 compact condition states, predictions, no duplicate pools'),domains=list(DOMAINS),variants=['V0_current','V1_fit_fixed_delta','V2_joint','V3_joint_no_omega','V4_sequence_oracle_delta'],cross_source=ev(ROOT/man['source_contract']['path']),parent_inputs=ev(AUDIT/'contract_v1.json'),author_recipe=ev(author.BASE/'author_recipe_v2/contract.json'),q='author cal8 q=.15, other current q=.05; reuse existing matched head, no new readout training',R_m=.6,clock_prior_seconds=.1,clock_Q_seconds=.005,clock_Q_origin='reuse peer07 diagnostic joint3_prior0; no val tuning',geometry='z from unchanged CV associations, correction applied to current heading-projected box centres; yaw unchanged; first-order ego displacement field',adoption='Author val21 mean AP70 V2-V0>=.20pp and sequence CI low>0; >=4/5 detector means positive; AP50 >=-.2pp. Missing cross domain/statistics => not adjudicable.',oracle='V4 is per-sequence evaluation-GT oracle clock, never deployment or selection; V1 supervised only on fit24',prior_formal='fixed-fit delta80.99ms and source-coefficient controls already have formal AP, not rerun under a new name')
    register('E3_temporal_baselines',__file__,'Separate persistence (hold-last) from smoothing (EMA/window/KF) under exactly the same CV-anchor measurements and readouts. EMA within .1pp of KF weakens a necessity claim, irrespective of signs.',dict(GPU_forwards=0,CPU_threads=1,wall_h='2-8 including calibration and both formal roles',disk_MiB='<120 compact scores/TP and state; shared geometry/pools referenced'),domain='author detector/current cal-q heading recipe',alpha_grid=[.1,.25,.5,.75,1.],window_messages=[3,5,10,20],selection='single calibration8 four-wait AP70 mean on matched U, tie smaller window / lower alpha; full val/hold never select',controls=['single_message_mean_zero_without_anchors','single_message_median_zero_without_anchors','hold_last_mean','EMA','window_anchor_weighted','KF'],readouts=['protected_detector_raw','same_frozen_matched_U'],remaining='RoCo/CBM/FreeAlign temporalization remains separately open, not claimed covered by these anchor baselines')
def load_author(role):
    if role in ('val21','mechanism_holdout14'):return author.materials(role)[:2]
    c=load(author.CONTRACT);d,_=author.formal.checked_dataset(ROOT/c['datasets']['mechanism_holdout14']);m=author.formal.materialize(d,role,np.zeros(2),np.zeros(2));g=[d.pool(int(i))['gt_boxes'] for i in m['dataset_index']];m['gt_boxes']=np.concatenate(g);m['gt_offsets']=np.r_[0,np.cumsum([len(a) for a in g])];return m,[d.meta[int(i)] for i in m['dataset_index']]
def materials(domain):
    if domain=='author':
        c=load(author.CONTRACT);head=ROOT/c['input_weight_identity']['geometry_heads']['kf']['weights']['path'];return {r:load_author(r) for r in ROLES},head.parent,.15
    if domain=='strict60':
        c=load(AUDIT/'contract_v1.json');result={}
        for r in ROLES:
            m=author.formal.load_npz(ROOT/c['inputs'][r+'_material']['path'])
            key='val_conditions' if r=='val21' else ('train_conditions' if r=='mechanism_holdout14' else 'fit_conditions')
            rows=[json.loads(l) for l in (ROOT/c['inputs'][key]['path']).read_text().splitlines()]
            by={(str(x['sequence']),int(x['receiver_timestamp_us']),int(x['wait_ms'])):x for x in rows}
            selected=[by[str(s),int(t),int(w)] for s,t,w in zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms'])];result[r]=(m,selected)
        return result,(ROOT/c['inputs']['matched_q005_head']['path']).parent,.05
    sc,man=cross_specs();mm,_,_=matched.prepare_domain(sc['domains'][domain]);result={}
    # Current receiver identity is shared, not the detector geometry or readout.
    # Join by timestamp/sequence/wait, never by a different domain's row index.
    from run_peer16_E1_timing_audit_v1_20261103 import POOL
    for role in ROLES:
        meta=[json.loads(l) for l in (POOL/('val_dataset' if role=='val21' else 'train_dataset')/'conditions.jsonl').read_text().splitlines()];by={(str(x['sequence']),int(x['receiver_timestamp_us']),int(x['wait_ms'])):x for x in meta};m=mm[role];rows=[by[str(s),int(t),int(w)] for s,t,w in zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms'])];assert all(r['role']==role for r in rows);result[role]=(m,rows)
    return result,(ROOT/man['domains'][domain]['matched_q005_head']['direct_u']['path']).parent,.05
def observation(m):
    obs=[]
    for a,b in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        a,b=int(a),int(b);remote=a+np.flatnonzero(m['source'][a:b]==1);anchor=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1));i,j=match(m['cv_boxes'][remote],m['cv_boxes'][anchor]);z=m['cv_boxes'][anchor[j],:2].astype(float)-m['cv_boxes'][remote[i],:2].astype(float);obs.append((remote[i],z))
    return obs
def temporal(m,obs,kind,q,parameter=None):
    states={};out=[]
    for seq,w,(ids,z) in zip(m['sequence'],m['wait_ms'],obs):
        key=str(seq),int(w);mean,var,hist=states.get(key,(np.zeros(2),1.,deque()));n=len(z);value=z.mean(0) if n else np.zeros(2)
        if kind=='KF':
            var+=q*q
            if n:k=var/(var+.36/n);mean=mean+k*(value-mean);var=(1-k)*var
        elif kind=='single_mean':mean=value
        elif kind=='single_median':mean=np.median(z,axis=0) if n else np.zeros(2)
        elif kind=='hold_last':
            if n:mean=value
        elif kind=='EMA':
            if n:mean=parameter*value+(1-parameter)*mean
        elif kind=='window':
            hist.append((n,z.sum(0) if n else np.zeros(2)))
            while len(hist)>parameter:hist.popleft()
            den=sum(x[0] for x in hist);mean=sum((x[1] for x in hist),np.zeros(2))/den if den else np.zeros(2)
        states[key]=(mean.copy(),var,hist);out.append(mean.copy())
    return np.asarray(out)
def evaluate(m,delta,head,name):
    panels,pred,_,_=author.evaluate(m,delta,head,name)
    boxes,_=author.heading.build_heading(m,delta)
    rawpan,rawpred=author.refit.fixed.base.evaluate(m,{name+'_raw':boxes},{name+'_raw':m['raw_scores']})
    pred.update(rawpred);return panels+rawpan,pred
def E3():
    out=BASE/'E3_temporal_baselines';start=time.monotonic();record('E3_temporal_baselines',dict(status='running_cal8_temporal_control_selection'))
    try:
        data,head,q=materials('author');m,_=data['calibration8'];obs=observation(m);cal=[]
        for kind,grid in [('EMA',[.1,.25,.5,.75,1.]),('window',[3,5,10,20])]:
            for value in grid:
                delta=temporal(m,obs,kind,q,value);pp,_,_,_=author.evaluate(m,delta,head,'cal_'+kind);cal.append(dict(kind=kind,value=value,AP70=float(np.mean([p['ap70'] for p in pp]))));atomic(out/'calibration_progress.json',cal)
        chosen={k:max([r for r in cal if r['kind']==k],key=lambda r:(r['AP70'],-r['value']))['value'] for k in ('EMA','window')};atomic(out/'calibration.json',dict(selected=chosen,all_candidates=cal,no_val_selection=True))
        for role in ROLES[2:]:
            m,rows=data[role];obs=observation(m);panels=[];preds={};states={}
            for kind in ('single_mean','single_median','hold_last','EMA','window','KF'):
                d=temporal(m,obs,kind,q,chosen.get(kind));pp,p=evaluate(m,d,head,kind);panels.extend(pp);preds.update(p);states[kind]=d;atomic(out/(role+'_panels.json'),panels)
                atomic(out/'status.json',dict(created_at=now(),status='formal_fixed_readout_temporal_baselines',role=role,method=kind,wall_s=time.monotonic()-start))
            preds.update(condition_sequence=m['sequence'],condition_GT_count=np.asarray([int(author.common_mask(m['gt_boxes'][int(a):int(b)]).sum()) for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]))
            np.savez_compressed(out/(role+'_predictions.npz'),**preds);np.savez_compressed(out/(role+'_states.npz'),**states)
        atomic(out/'terminal.json',dict(created_at=now(),status='complete_light_checked',wall_s=time.monotonic()-start,GPU_forwards=0,bootstrap_and_external_temporalization_pending=True));record('E3_temporal_baselines',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record('E3_temporal_baselines',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise
def clock_fits(m,rows,obs,oracle=False):
    dyn=dynamics(rows);groups=defaultdict(lambda:([],[]))
    for i,(r,(ids,z)) in enumerate(zip(rows,obs)):
        v,w,ok=dyn[str(r['sequence']),int(r['receiver_timestamp_us'])]
        if not ok:continue
        if oracle:
            a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2]);ri=a+np.flatnonzero(m['source'][a:b]==1);ii,jj=match(m['cv_boxes'][ri],m['gt_boxes'][ga:gb],3.);ids=ri[ii];z=m['gt_boxes'][ga:gb][jj,:2].astype(float)-m['cv_boxes'][ids,:2].astype(float)
        if len(z):
            p=m['cv_boxes'][ids,:2];terms=v[None,:]+w*np.column_stack((-p[:,1],p[:,0]));key=str(r['sequence']) if oracle else 'fit24';groups[key][0].append(z);groups[key][1].append(terms)
    return {k:fit(np.concatenate(e),np.concatenate(mm)) for k,(e,mm) in groups.items()}
def joint_geometry(m,rows,obs,variant,q,fixed,oracle):
    dyn=dynamics(rows);states={};deltas=[];delta_all=[];clock=[]
    boxes=m['cv_boxes'].copy();features=m['cv_features'].copy()
    base,_=author.heading.build_heading(m,np.zeros((len(rows),2)))
    boxes[:]=base
    for i,(r,(ids,z)) in enumerate(zip(rows,obs)):
        key=str(r['sequence']),int(r['wait_ms']);v,w,ok=dyn[str(r['sequence']),int(r['receiver_timestamp_us'])]
        if variant=='V3_joint_no_omega':w=0.
        a,b=map(int,m['condition_offsets'][i:i+2]);remote=a+np.flatnonzero(m['source'][a:b]==1);anchor=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1));x,P=states.get(key,(np.zeros(3),np.diag([1.,1.,.1**2])));x=x.copy();P=P+np.diag([q*q,q*q,.005**2]);p=m['cv_boxes'][ids,:2];terms=v[None,:]+w*np.column_stack((-p[:,1],p[:,0]))
        if variant in ('V1_fit_fixed_delta','V4_sequence_oracle_delta'):
            delta=fixed if variant=='V1_fit_fixed_delta' else oracle.get(str(r['sequence']),0.)
            if len(z):
                gain=P[0,0]/(P[0,0]+.36/len(z));x[:2]+=gain*((z+delta*terms).mean(0)-x[:2]);P[0,0]*=1-gain;P[1,1]=P[0,0]
            x[2]=delta
        else:
            for zz,term in zip(z,terms):
                H=np.array([[1.,0.,-term[0]],[0.,1.,-term[1]]]);R=.36*np.eye(2);K=np.linalg.solve(H@P@H.T+R,H@P).T;x+=K@(zz-H@x);I=np.eye(3)-K@H;P=I@P@I.T+K@R@K.T;P=(P+P.T)/2
        p=m['cv_boxes'][remote,:2];term=v[None,:]+w*np.column_stack((-p[:,1],p[:,0]));corr=x[:2][None,:]-x[2]*term;boxes[remote,:2]+=corr.astype(np.float32)
        features[remote]=author.formal.update_features(m['cv_features'][remote],boxes[remote],m['cv_boxes'][anchor],m['raw_scores'][anchor]);states[key]=(x,P);clock.append(x.copy())
    return boxes,features,np.asarray(clock)
def E2(domain):
    out=BASE/'E2_comet_delta'/domain;out.mkdir(exist_ok=False);start=time.monotonic();name='E2_'+domain;record(name,dict(status='running_own_input_joint_clock_state'))
    try:
        data,head,q=materials(domain);fitm,fitrows=data['fit24'];obsfit=observation(fitm);fits=clock_fits(fitm,fitrows,obsfit,True);fixed=fits['fit24']['delta_ms']/1000;atomic(out/'fit24_clock.json',dict(fits=fits,supervised_fit_only=True,q=q,head=ev(head/'direct_u.pth'),no_evaluation_selection=True))
        for role in ROLES[2:]:
            m,rows=data[role];obs=observation(m);n=np.asarray([len(z) for _,z in obs]);means=np.asarray([z.mean(0) if len(z) else np.zeros(2) for _,z in obs]);delta,_,_,_=author.kf(m,n,means,q,.6);panels,pred,_,_=author.evaluate(m,delta,head,'V0_current');rawpan,rawpred=author.refit.fixed.base.evaluate(m,{'V0_current_raw':author.heading.build_heading(m,delta)[0]},{'V0_current_raw':m['raw_scores']});panels+=rawpan;pred.update(rawpred)
            oraclefit=clock_fits(m,rows,obs,True);oracle={s:v['delta_ms']/1000 for s,v in oraclefit.items() if v};atomic(out/(role+'_oracle_clock.json'),oraclefit);states={}
            for var in ('V1_fit_fixed_delta','V2_joint','V3_joint_no_omega','V4_sequence_oracle_delta'):
                b,f,x=joint_geometry(m,rows,obs,var,q,fixed,oracle);score=author.refit.score_with_head(m,f,head);pp,p=author.refit.fixed.base.evaluate(m,{var:b,var+'_raw':b},{var:score,var+'_raw':m['raw_scores']});panels+=pp;pred.update(p);states[var]=x;atomic(out/(role+'_panels.json'),panels);atomic(out/'status.json',dict(created_at=now(),status='formal_joint_clock_state',role=role,variant=var,wall_s=time.monotonic()-start))
            pred.update(condition_sequence=m['sequence'],condition_GT_count=np.asarray([int(author.common_mask(m['gt_boxes'][int(a):int(b)]).sum()) for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]))
            np.savez_compressed(out/(role+'_predictions.npz'),**pred);np.savez_compressed(out/(role+'_states.npz'),**states)
        atomic(out/'terminal.json',dict(created_at=now(),status='complete_light_checked',domain=domain,GPU_forwards=0,wall_s=time.monotonic()-start,adoption_not_yet_adjudicable=True,bootstrap_independent_and_source_distance_analysis_pending=True));record(name,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','E3',*DOMAINS]);a=p.parse_args();prepare() if a.mode=='prepare' else E3() if a.mode=='E3' else E2(a.mode)
