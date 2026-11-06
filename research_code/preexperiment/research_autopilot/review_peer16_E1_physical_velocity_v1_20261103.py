"""Independent public-pose/ICP join and independently implemented cluster CI.

Reuses original label-free physical registration; never re-register pointclouds.
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from peer16_v1_20261103 import ROOT,AUDIT,BASE,atomic,ev,load,now,register,record
from run_peer16_E1_timing_audit_v1_20261103 import POOL
OUT=BASE/'E1_physical_velocity_independent'
REG=AUDIT/'cycle21_peer14_v1_20261102/raw_registration_v1/producer'

def prepare():
    register('E1_physical_velocity_independent',__file__,
        'The old E1 file already contains ICP-speed correlations; its missing '
        'status was stale. Independently rebuild causal velocities from '
        'public-pose rows and join immutable ICP by sequence/frame, check '
        'all8 old point estimates and add direct whole-sequence scipy '
        'bootstrap CI. A |r|>.3 proxy gate is not proof of unique physical '
        'clock causality; frozen quality subset and all-frame panels both retained.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='<1',disk_MiB='<3'),
        registration=ev(REG/'terminal.json'),quality=ev(REG/'quality_frozen.json'),
        previous=ev(BASE/'E1_timing_audit/ICP_velocity_panels.json'),
        resamples=2000,seed=2026110321,unit='whole_sequence')

def statistic(seq,x,y):
    seq=np.asarray(seq);x=np.asarray(x,float);y=np.asarray(y,float)
    names=np.unique(seq);groups=[np.flatnonzero(seq==s) for s in names]
    point=float(pearsonr(x,y).statistic);rng=np.random.default_rng(2026110321);values=[]
    for _ in range(2000):
        ids=np.concatenate([groups[int(k)] for k in rng.integers(len(names),size=len(names))])
        if np.ptp(x[ids])>1e-12 and np.ptp(y[ids])>1e-12:
            values.append(float(pearsonr(x[ids],y[ids]).statistic))
    return dict(r=point,CI95=np.quantile(values,[.025,.975]).tolist(),
        valid_resamples=len(values),sequences=len(names),frames=len(x))

def run():
    start=time.monotonic();record('E1_physical_velocity_independent',dict(status='running_independent_pose_ICP_velocity_join'))
    try:
        grouped=defaultdict(dict)
        for ds in ('train_dataset','val_dataset'):
            for line in (POOL/ds/'conditions.jsonl').open():
                r=json.loads(line)
                if int(r['wait_ms'])!=0:continue
                key=str(r['sequence']);t=int(r['receiver_timestamp_us'])
                if t in grouped[key]:assert grouped[key][t]['current_vehicle_frame']==r['current_vehicle_frame']
                grouped[key][t]=r
        motion={}
        for seq,rows in grouped.items():
            times=sorted(rows)
            for prev,t in zip(times,times[1:]):
                dt=(t-prev)/1e6
                if not 0<dt<=.35:continue
                a=np.asarray(rows[prev]['current_ego_pose_lidar_to_world']);b=np.asarray(rows[t]['current_ego_pose_lidar_to_world'])
                velocity=b[:2,:2].T.dot(b[:2,3]-a[:2,3])/dt
                yaw=np.arctan2(b[1,0],b[0,0])-np.arctan2(a[1,0],a[0,0])
                omega=np.arctan2(np.sin(yaw),np.cos(yaw))/dt
                motion[seq,str(rows[t]['current_vehicle_frame'])]=(velocity[0],omega)
        accepted=set(load(REG/'accepted_indices.json'));values=defaultdict(list)
        for line in (REG/'registrations.jsonl').open():
            r=json.loads(line);key=str(r['sequence']),str(r['frame'])
            if r['dataset']!='SPD' or r.get('error') or key not in motion:continue
            v,w=motion[key];item=(key[0],v,float(r['delta_xy'][0]),w,float(r['yaw']))
            values[r['role'],'all'].append(item)
            if int(r['index']) in accepted:values[r['role'],'fit_quality_subset'].append(item)
        old={(r['role'],r['quality']):r for r in load(BASE/'E1_timing_audit/ICP_velocity_panels.json')}
        panels=[]
        for (role,quality),items in sorted(values.items()):
            seq=[r[0] for r in items];a=np.asarray([r[1:] for r in items])
            sx=statistic(seq,a[:,0],a[:,1]);sy=statistic(seq,a[:,2],a[:,3])
            target=old[role,quality]
            assert target['frames']==len(items) and abs(target['ICP_x_velocity_r']-sx['r'])<1e-12
            assert abs(target['ICP_yaw_yawrate_r']-sy['r'])<1e-12
            panels.append(dict(role=role,quality=quality,x_speed=sx,yaw_rate=sy,
                proxy_gate=abs(sx['r'])>.3,old_points_exact=True,
                physical_ground_truth=False,unique_clock_cause=False))
            atomic(OUT/'panels.json',panels)
        assert len(panels)==8
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
            panels=ev(OUT/'panels.json'),GPU_forwards=0,wall_s=time.monotonic()-start,
            independent_statistical_implementation='Direct sequence resampling and scipy pearsonr, not producer moment bootstrap',
            scope='Independent ICP-speed/public-motion aggregation, not an independent raw ICP solver or clock synchronization measurement'))
        record('E1_physical_velocity_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('E1_physical_velocity_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
