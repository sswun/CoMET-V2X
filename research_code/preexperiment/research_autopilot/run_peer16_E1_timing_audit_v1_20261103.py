"""Recompute timing evidence from original public poses and detector pools."""
import argparse,json,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from peer16_v1_20261103 import ROOT,AUTO,AUDIT,BASE,atomic,ev,load,now,register,record
OUT=BASE/'E1_timing_audit'
POOL=AUTO/'status_audit_20261021_1200/object_history_selector_v1'
DATA=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'
def prepare():
    register('E1_timing_audit',__file__,'Recompute label-free anchor/velocity timing correlation, per-sequence delay identifiability and raw-ICP relation. A geometric correlation alone does not prove clock causality. Keep fit/cal/hold/val separate, all waits, include zero-motion/missing-history limitations.',dict(GPU_forwards=0,CPU_threads=1,wall_h='1-4',disk_MiB='<20 diagnostic arrays only; original pools and ICP referenced'),pool_summaries=[ev(POOL/x/'summary.json') for x in ('train_dataset','val_dataset')],scan_ms=list(range(-200,201,10)),preregistered='Per-sequence delay CV<.30 for clock-bias interpretation; |ICP correlation|>.30 only on fixed fit-quality subset. Label-assisted source attribution remains separate.',prior_formal_timing=ev(ROOT/'docs/review_20261031/traf_timing_structured_observation_fitcal_v2_results.md'))
def dynamics(rows):
    grouped=defaultdict(dict)
    for r in rows:grouped[str(r['sequence'])][int(r['receiver_timestamp_us'])]=np.asarray(r['current_ego_pose_lidar_to_world'],float)
    result={}
    for seq,d in grouped.items():
        ts=sorted(d)
        for k,t in enumerate(ts):
            v=np.zeros(2);w=0.;ok=False
            if k:
                dt=(t-ts[k-1])/1e6
                if 0<dt<=.35:
                    T,P=d[t],d[ts[k-1]];v=T[:2,:2].T@(T[:2,3]-P[:2,3])/dt
                    yaw=np.arctan2(T[1,0],T[0,0])-np.arctan2(P[1,0],P[0,0]);w=float(np.arctan2(np.sin(yaw),np.cos(yaw))/dt);ok=True
            result[seq,t]=(v,w,ok)
    return result
def match(a,b,gate=2.5):
    if not len(a) or not len(b):return np.empty(0,int),np.empty(0,int)
    d=np.linalg.norm(a[:,None,:2]-b[None,:,:2],axis=-1);i,j=linear_sum_assignment(d)
    good=(d[i,j]<=gate)&(d.argmin(1)[i]==j)&(d.argmin(0)[j]==i)
    return i[good],j[good]
def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])
def fit(e,m):
    if len(e)<3:return None
    A=np.zeros((2*len(e),3));A[0::2,0]=1;A[1::2,1]=1;A[:,2]=-m.ravel()
    x,_,rank,s=np.linalg.lstsq(A,e.ravel(),rcond=None)
    return dict(c=x[:2].tolist(),delta_ms=float(1000*x[2]),rank=int(rank),condition_number=float(s[0]/max(s[-1],1e-12)),rows=len(e))
def matrix(path,nested=False):
    d=load(path);d=d['transform'] if nested else d;T=np.eye(4);T[:3,:3]=d['rotation'];T[:3,3]=np.asarray(d['translation']).ravel();return T
def run():
    start=time.monotonic();cpu=time.process_time();record('E1_timing_audit',dict(status='running_original_pool_pose_anchor_audit'))
    try:
        info={str(r['frame_id']):r for r in load(DATA/'vehicle-side/data_info.json')};max_pose=0.;timegaps=[];summaries=[];lookup={};allrows=[]
        for dsname in ('train_dataset','val_dataset'):
            root=POOL/dsname;meta=[json.loads(l) for l in (root/'conditions.jsonl').read_text().splitlines()];dyn=dynamics(meta);allrows.extend(meta)
            with np.load(root/'pools.npz') as z:boxes=z['boxes'];kind=z['hypothesis_type'];prot=z['is_protected'];ids=z['source_detection_indices']
            grouped=defaultdict(list)
            for ordinal,r in enumerate(meta):
                seq=str(r['sequence']);t=int(r['receiver_timestamp_us']);v,w,ok=dyn[seq,t]
                if int(r['wait_ms'])==0:
                    d=info[str(r['current_vehicle_frame'])];T=matrix(DATA/'vehicle-side'/d['calib_novatel_to_world_path'])@matrix(DATA/'vehicle-side'/d['calib_lidar_to_novatel_path'],True)
                    max_pose=max(max_pose,float(np.max(abs(T-np.asarray(r['current_ego_pose_lidar_to_world'])))))
                    gap=(int(d['image_timestamp'])-int(d['pointcloud_timestamp']))/1e6 if 'image_timestamp' in d else None
                    if gap is not None:timegaps.append(dict(role=r['role'],sequence=seq,gap_ms=1000*gap,speed=float(np.linalg.norm(v)),vx_gap=float(v[0]*gap)))
                    lookup[seq,str(r['current_vehicle_frame'])]=(v,w,ok,r['role'])
                if not ok:continue
                a,b=int(r['start']),int(r['stop']);k=kind[a:b];ix=np.arange(a,b);remote=ix[k==1];prop={int(ids[j]):j for j in ix[k==2]}
                cv=boxes[[prop.get(int(ids[j]),j) for j in remote]] if len(remote) else np.empty((0,7));ego=boxes[ix[(k==0)&(prot[a:b]==1)]]
                ri,ei=match(cv,ego);e=ego[ei,:2].astype(float)-cv[ri,:2].astype(float)
                motion=v[None,:]+w*np.column_stack((-cv[ri,1],cv[ri,0]))
                for ee,mm in zip(e,motion):grouped[r['role'],int(r['wait_ms'])].append((seq,ee,mm,v,w))
                if ordinal%1024==0:atomic(OUT/'status.json',dict(created_at=now(),status='label_free_anchor_audit',dataset=dsname,conditions=ordinal+1))
            for (role,wait),rows in grouped.items():
                seq=np.asarray([r[0] for r in rows]);e=np.stack([r[1] for r in rows]);m=np.stack([r[2] for r in rows]);v=np.stack([r[3] for r in rows]);w=np.asarray([r[4] for r in rows]);per=[];ec=e.copy();vc=v.copy()
                for s in sorted(set(seq)):
                    use=seq==s;f=fit(e[use],m[use]);per.append(dict(sequence=s,fit=f));ec[use]-=e[use].mean(0);vc[use]-=v[use].mean(0)
                values=[x['fit']['delta_ms'] for x in per if x['fit'] and x['fit']['rank']==3 and x['fit']['condition_number']<1e6]
                cv=float(np.std(values,ddof=1)/max(abs(np.mean(values)),1e-12)) if len(values)>1 else None
                scan=[]
                for dt in range(-200,201,10):
                    adjusted=e+dt/1000*m;bias=adjusted.mean(0);scan.append(dict(delta_ms=dt,RMS_after_global_constant=float(np.sqrt(np.mean((adjusted-bias)**2)))))
                bins=[];speed=np.linalg.norm(v,axis=1)
                for lo,hi in ((.5,3),(3,6),(6,10),(10,100)):
                    use=(speed>=lo)&(speed<hi);bins.append(dict(speed_min=lo,speed_max=hi,anchors=int(use.sum()),residual_x_mean=float(e[use,0].mean()) if use.any() else None))
                summaries.append(dict(role=role,wait_ms=wait,anchors=len(rows),fit=fit(e,m),longitudinal_correlation=corr(v[:,0],e[:,0]),within_sequence_longitudinal_correlation=corr(vc[:,0],ec[:,0]),lateral_yawrate_correlation=corr(w,e[:,1]),per_sequence=per,per_sequence_delay_CV=cv,clock_stability_gate=cv is not None and cv<.3,speed_bins=bins,pose_scan=scan,scan_not_method_selection=True))
                np.savez_compressed(OUT/f'{role}_wait{wait:03d}_anchors.npz',sequence=seq,residual=e,motion=m,velocity=v,yawrate=w)
        atomic(OUT/'anchor_timing_panels.json',summaries)
        atomic(OUT/'public_pose_and_timestamp_audit.json',dict(max_abs_pose_error=max_pose,exact_dataset_same_frame=max_pose<1e-10,image_minus_pointcloud=timegaps,causal_velocity='backward pose differences, gap<=.35s; first frame has no motion estimate'))
        reg=AUDIT/'cycle21_peer14_v1_20261102/raw_registration_v1/producer';physical=[]
        if (reg/'registrations.jsonl').exists():
            accepted=set(load(reg/'accepted_indices.json'));group=defaultdict(list)
            for line in (reg/'registrations.jsonl').read_text().splitlines():
                r=json.loads(line);key=str(r['sequence']),str(r['frame'])
                if r['dataset']!='SPD' or r.get('error') or key not in lookup:continue
                v,w,ok,role=lookup[key]
                if ok:group[role,'all'].append((v[0],r['delta_xy'][0],w,r['yaw']))
                if ok and r['index'] in accepted:group[role,'fit_quality_subset'].append((v[0],r['delta_xy'][0],w,r['yaw']))
            for (role,quality),r in group.items():
                a=np.asarray(r);rx=corr(a[:,0],a[:,1]);physical.append(dict(role=role,quality=quality,frames=len(a),ICP_x_velocity_r=rx,ICP_yaw_yawrate_r=corr(a[:,2],a[:,3]),physical_proxy_gate=rx is not None and abs(rx)>.3,not_physical_ground_truth=True))
        atomic(OUT/'ICP_velocity_panels.json',physical)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',anchor_panels=ev(OUT/'anchor_timing_panels.json'),public_pose=ev(OUT/'public_pose_and_timestamp_audit.json'),ICP=ev(OUT/'ICP_velocity_panels.json'),GPU_forwards=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,not_independent_replication_of_all_peer_claims=True,remaining=['veh_coop GT-only sequence delay distribution','roadside clock slope and timestamp attribution','cluster bootstrap on timing correlations'],no_model_wakeup=True));record('E1_timing_audit',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record('E1_timing_audit',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
